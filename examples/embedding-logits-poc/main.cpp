#include "llama.h"
#include "ggml-backend.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>
#include <unistd.h>
#ifdef __APPLE__
#include <mach/mach.h>
#endif

using Clock = std::chrono::steady_clock;
static const auto epoch = Clock::now();
static double ms(Clock::duration d) { return std::chrono::duration<double, std::milli>(d).count(); }
static double elapsed() { return ms(Clock::now() - epoch); }
static std::string quote(const std::string & s) {
    std::ostringstream o;
    o << '"';
    for (unsigned char c : s) {
        if (c == '"' || c == '\\') o << '\\' << c;
        else if (c < 32) o << "\\u" << std::hex << std::setw(4) << std::setfill('0') << int(c) << std::dec;
        else o << c;
    }
    o << '"';
    return o.str();
}
static int number(const std::string & s, bool zero = false) {
    size_t used = 0;
    const auto n = std::stoll(s, &used);
    if (used != s.size() || n < (zero ? 0 : 1) || n > 1000000) throw std::runtime_error("invalid integer: " + s);
    return int(n);
}
struct Options {
    std::string model, device = "cpu", output;
    int tokens = 256, contexts = 1, iterations = 3, warmup = 1;
    std::vector<int> lengths;
};
static Options options(int argc, char ** argv) {
    Options o;
    for (int i = 1; i < argc; ++i) {
        const std::string k = argv[i];
        if (k == "--help") {
            std::cout << "--model FILE --device cpu|metal|cuda --tokens 256 --contexts 1 --iterations N --warmup N --lengths 32,256,257 --output FILE\n";
            std::exit(0);
        }
        if (++i == argc) throw std::runtime_error("missing value for " + k);
        const std::string v = argv[i];
        if (k == "--model") o.model = v;
        else if (k == "--device") o.device = v;
        else if (k == "--output") o.output = v;
        else if (k == "--tokens") o.tokens = number(v);
        else if (k == "--contexts") o.contexts = number(v);
        else if (k == "--iterations") o.iterations = number(v);
        else if (k == "--warmup") o.warmup = number(v, true);
        else if (k == "--lengths") {
            std::istringstream parts(v);
            std::string part;
            while (std::getline(parts, part, ',')) o.lengths.push_back(number(part));
            if (o.lengths.empty()) throw std::runtime_error("empty --lengths");
        } else throw std::runtime_error("unknown option: " + k);
    }
    if (o.model.empty() || o.output.empty()) throw std::runtime_error("--model and --output are required");
    if (o.device != "cpu" && o.device != "metal" && o.device != "cuda") throw std::runtime_error("invalid device");
    if (o.contexts > 128) throw std::runtime_error("contexts must be <= 128");
    if (o.lengths.empty()) o.lengths.push_back(o.tokens);
    return o;
}

struct Memory {
    double time_ms = elapsed();
    uint64_t rss = 0, footprint = 0, hwm = 0;
    bool valid = false;
};
static Memory memory() {
    Memory m;
#ifdef __APPLE__
    task_vm_info_data_t info{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) == KERN_SUCCESS) {
        m.rss = info.resident_size;
        m.footprint = info.phys_footprint;
        m.hwm = info.resident_size_peak;
        m.valid = true;
    }
#else
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream fields(line);
        std::string name;
        uint64_t kb = 0;
        fields >> name >> kb;
        if (name == "VmRSS:") { m.rss = kb * 1024; m.valid = true; }
        if (name == "VmHWM:") m.hwm = kb * 1024;
    }
#endif
    return m;
}
static void write_memory(std::ostream & o, const Memory & m) {
    o << "{\"time_ms\":" << m.time_ms << ",\"valid\":" << (m.valid ? "true" : "false")
      << ",\"rss_bytes\":" << m.rss << ",\"physical_footprint_bytes\":";
#ifdef __APPLE__
    o << m.footprint;
#else
    o << "null";
#endif
    o << ",\"rss_high_water_bytes\":" << m.hwm << '}';
}
struct Monitor {
    std::atomic<bool> stopping{false};
    std::vector<Memory> samples;
    std::vector<std::pair<std::string, Memory>> stages;
    std::thread thread;
    Monitor() : thread([this] {
        while (!stopping.load()) {
            samples.push_back(memory());
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    }) {}
    void stage(const std::string & label) { stages.emplace_back(label, memory()); }
    void stop() { stopping.store(true); if (thread.joinable()) thread.join(); }
    ~Monitor() { stop(); }
    void write(std::ostream & o) {
        stop();
        uint64_t rss = 0, footprint = 0;
        for (const auto & m : samples) { rss = std::max(rss, m.rss); footprint = std::max(footprint, m.footprint); }
        for (const auto & s : stages) { rss = std::max(rss, s.second.rss); footprint = std::max(footprint, s.second.footprint); }
        o << "{\"sampling_interval_ms\":20,\"peak_sampled_rss_bytes\":" << rss << ",\"peak_sampled_physical_footprint_bytes\":";
#ifdef __APPLE__
        o << footprint;
#else
        o << "null";
#endif
        o << ",\"stages\":[";
        for (size_t i = 0; i < stages.size(); ++i) {
            if (i) o << ',';
            o << "{\"stage\":" << quote(stages[i].first) << ",\"memory\":";
            write_memory(o, stages[i].second); o << '}';
        }
        o << "],\"samples\":[";
        for (size_t i = 0; i < samples.size(); ++i) { if (i) o << ','; write_memory(o, samples[i]); }
        o << "]}";
    }
};

struct Input { int length; std::string text; std::vector<llama_token> tokens; };
static std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    int n = llama_tokenize(vocab, text.data(), int(text.size()), nullptr, 0, true, false);
    if (n == std::numeric_limits<int32_t>::min()) throw std::runtime_error("tokenizer overflow");
    std::vector<llama_token> tokens(size_t(std::abs(n)));
    n = llama_tokenize(vocab, text.data(), int(text.size()), tokens.data(), int(tokens.size()), true, false);
    if (n < 0) throw std::runtime_error("tokenize failed");
    tokens.resize(size_t(n));
    return tokens;
}
// Search text repetition counts, never truncate token IDs or remove special tokens.
static Input make_input(const llama_vocab * vocab, int target) {
    const std::vector<std::string> words = {"search", "store", "read", "update", "check", "process", "share", "return"};
    int low = 0, high = target * 2 + 16;
    while (low <= high) {
        const int count = low + (high - low) / 2;
        std::string text;
        for (int i = 0; i < count; ++i) {
            if (i) text += ' ';
            text += words[size_t(i) % words.size()];
        }
        auto tokens = tokenize(vocab, text);
        if (int(tokens.size()) == target) return {target, std::move(text), std::move(tokens)};
        if (int(tokens.size()) < target) low = count + 1;
        else high = count - 1;
    }
    for (const std::string unit : {" a", " x", " code", " test", " vector", " 0"}) {
        int lo = 0, hi = target * 2 + 16;
        while (lo <= hi) {
            const int repeats = lo + (hi - lo) / 2;
            std::string text = "a";
            for (int i = 0; i < repeats; ++i) text += unit;
            auto tokens = tokenize(vocab, text);
            if (int(tokens.size()) == target) return {target, std::move(text), std::move(tokens)};
            if (int(tokens.size()) < target) lo = repeats + 1;
            else hi = repeats - 1;
        }
    }
    throw std::runtime_error("cannot construct exact token length with intact special tokens: " + std::to_string(target));
}
static std::string meta(const llama_model * model, const char * key) {
    char value[256];
    const int n = llama_model_meta_val_str(model, key, value, sizeof(value));
    if (n < 0 || n >= int(sizeof(value))) throw std::runtime_error(std::string("missing/oversize model metadata: ") + key);
    return std::string(value, size_t(n));
}
static ggml_backend_dev_t select_device(const std::string & device) {
    if (device == "cpu") return nullptr;
    for (size_t i = 0; i < ggml_backend_dev_count(); ++i) {
        auto dev = ggml_backend_dev_get(i);
        std::string name = ggml_backend_dev_name(dev);
        for (char & c : name) c = char(std::tolower(static_cast<unsigned char>(c)));
        if ((device == "metal" && (name.find("metal") != std::string::npos || name.find("mtl") != std::string::npos)) ||
            (device == "cuda" && name.find("cuda") != std::string::npos)) return dev;
    }
    throw std::runtime_error("requested GPU backend is not available: " + device);
}
using Context = std::unique_ptr<llama_context, decltype(&llama_free)>;
struct Slot {
    Context ctx{nullptr, llama_free};
    uint32_t capacity = 0;
    void ensure(llama_model * model, const std::string & arch, int tokens, int limit, bool gpu) {
        if (capacity >= uint32_t(tokens)) return;
        uint32_t next = 32;
        while (next < uint32_t(tokens)) next *= 2;
        next = std::min(next, uint32_t(limit));
        ctx.reset();
        auto p = llama_context_default_params();
        p.n_ctx = arch == "qwen3" ? std::min(uint32_t(limit), std::max(256u, next)) : uint32_t(limit);
        p.n_batch = next;
        p.n_ubatch = arch == "gemma-embedding" ? next : std::min(256u, next);
        p.n_seq_max = 1;
        p.n_threads = 1;
        p.n_threads_batch = 1;
        p.embeddings = true;
        p.kv_unified = true;
        p.offload_kqv = gpu;
        p.op_offload = gpu;
        ctx.reset(llama_init_from_model(model, p));
        if (!ctx) throw std::runtime_error("context creation failed");
        if (llama_pooling_type(ctx.get()) == LLAMA_POOLING_TYPE_NONE) throw std::runtime_error("pooled embeddings required");
        capacity = next;
    }
};
struct Job {
    double latency_ms = 0;
    Clock::time_point end;
    std::vector<float> embedding;
    std::string error;
};
static Job infer(llama_context * context, const Input & input, int width) {
    Job job;
    const auto start = Clock::now();
    try {
        if (auto mem = llama_get_memory(context)) llama_memory_clear(mem, true);
        auto batch = llama_batch_init(input.length, 0, 1);
        struct BatchGuard { llama_batch & b; ~BatchGuard() { llama_batch_free(b); } } guard{batch};
        batch.n_tokens = input.length;
        for (int i = 0; i < input.length; ++i) {
            batch.token[i] = input.tokens[size_t(i)];
            batch.pos[i] = i;
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = 0;
            batch.logits[i] = true;
        }
        const int code = llama_decode(context, batch);
        if (code != 0) throw std::runtime_error("llama_decode returned " + std::to_string(code));
        const float * values = llama_get_embeddings_seq(context, 0); // Synchronizes decode and transfers.
        if (!values) throw std::runtime_error("missing sequence embedding");
        job.embedding.assign(values, values + width);
        for (float value : job.embedding) if (!std::isfinite(value)) throw std::runtime_error("nonfinite embedding");
    } catch (const std::exception & e) { job.error = e.what(); }
    job.end = Clock::now();
    job.latency_ms = ms(job.end - start);
    return job;
}
// Persistent threads; the round timer starts only after input and contexts are ready.
class Team {
    std::vector<Slot> & slots;
    int width;
    std::mutex mutex;
    std::condition_variable changed;
    bool stopping = false;
    size_t generation = 0, ready = 0, done = 0;
    const Input * input = nullptr;
    std::vector<std::thread> threads;
public:
    std::vector<Job> jobs;
    Team(std::vector<Slot> & s, int w) : slots(s), width(w), jobs(s.size()) {
        for (size_t i = 0; i < slots.size(); ++i) threads.emplace_back([this, i] {
            std::unique_lock<std::mutex> lock(mutex);
            size_t seen = generation;
            ++ready; changed.notify_all();
            while (true) {
                changed.wait(lock, [&] { return stopping || generation != seen; });
                if (stopping) return;
                seen = generation;
                const Input * current = input;
                lock.unlock();
                jobs[i] = infer(slots[i].ctx.get(), *current, width);
                lock.lock();
                ++done; changed.notify_all();
            }
        });
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&] { return ready == slots.size(); });
    }
    double run(const Input & current) {
        std::unique_lock<std::mutex> lock(mutex);
        input = &current;
        done = 0;
        const auto start = Clock::now();
        ++generation;
        changed.notify_all();
        changed.wait(lock, [&] { return done == slots.size(); });
        auto end = start;
        for (const auto & job : jobs) {
            if (!job.error.empty()) throw std::runtime_error(job.error);
            end = std::max(end, job.end);
        }
        return ms(end - start);
    }
    ~Team() {
        { std::lock_guard<std::mutex> lock(mutex); stopping = true; }
        changed.notify_all();
        for (auto & t : threads) t.join();
    }
};

int main(int argc, char ** argv) {
    try {
        const auto o = options(argc, argv);
        const char * mode_env = std::getenv("POC_LOGITS_MODE");
        const std::string mode = mode_env ? mode_env : "baseline";
        if (mode != "baseline" && mode != "host_off" && mode != "full_off") throw std::runtime_error("invalid POC_LOGITS_MODE");
        std::ofstream out(o.output);
        if (!out) throw std::runtime_error("cannot open output");
        out << std::setprecision(17);
        Monitor monitor;
        monitor.stage("start");
        const char * backend_dir = std::getenv("POC_BACKEND_DIR");
        if (backend_dir && *backend_dir) {
            const char * extra_backend = std::getenv("GGML_BACKEND_PATH");
            if (extra_backend && *extra_backend) throw std::runtime_error("unset GGML_BACKEND_PATH when using POC_BACKEND_DIR");
            ggml_backend_load_all_from_path(backend_dir);
        } else {
            ggml_backend_load_all();
        }
        llama_backend_init();
        struct BackendGuard { ~BackendGuard() { llama_backend_free(); } } backend;
        auto selected = select_device(o.device);
        ggml_backend_dev_t devices[] = {selected, nullptr};
        auto params = llama_model_default_params();
        params.devices = devices; // An empty list explicitly selects CPU-only model placement.
        params.n_gpu_layers = selected ? -1 : 0;
        params.split_mode = LLAMA_SPLIT_MODE_NONE;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(o.model.c_str(), params), llama_model_free);
        if (!model) throw std::runtime_error("model loading failed");
        monitor.stage("model_loaded");
        const std::string arch = meta(model.get(), "general.architecture");
        if (arch != "gemma-embedding" && arch != "qwen3") throw std::runtime_error("only gemma-embedding and qwen3 supported");
        const int limit = std::min(arch == "qwen3" ? 8192 : 2048, llama_model_n_ctx_train(model.get()));
        const auto * vocab = llama_model_get_vocab(model.get());
        const int width = llama_model_n_embd_out(model.get());
        if (width <= 0) throw std::runtime_error("invalid embedding width");
        std::vector<Input> inputs;
        for (int length : o.lengths) {
            if (length > limit) throw std::runtime_error("length exceeds context limit");
            inputs.push_back(make_input(vocab, length));
        }
        char hostname[256]{};
        gethostname(hostname, sizeof(hostname) - 1);
        out << "{\"schema\":1,\"mode\":" << quote(mode) << ",\"native_revision\":\"5f55650a78f92aff4d48d671423e888fac0469ff\",\"host\":" << quote(hostname)
            << ",\"model\":" << quote(o.model) << ",\"architecture\":" << quote(arch) << ",\"device\":" << quote(o.device)
            << ",\"selected_backend\":" << quote(selected ? ggml_backend_dev_name(selected) : "CPU")
            << ",\"backend_directory\":" << quote(backend_dir ? backend_dir : "")
            << ",\"contexts\":" << o.contexts << ",\"threads_per_context\":1,\"iterations\":" << o.iterations
            << ",\"warmup\":" << o.warmup << ",\"vocab_size\":" << llama_vocab_n_tokens(vocab) << ",\"embedding_width\":" << width
            << ",\"tokenizer_add_special\":true,\"tokenizer_parse_special\":false,\"persistent_workers\":true,\"inputs\":[";
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i) out << ',';
            out << "{\"length\":" << inputs[i].length << ",\"text\":" << quote(inputs[i].text) << ",\"token_ids\":[";
            for (size_t t = 0; t < inputs[i].tokens.size(); ++t) { if (t) out << ','; out << inputs[i].tokens[t]; }
            out << "]}";
        }
        std::vector<Slot> slots(size_t(o.contexts));
        for (auto & slot : slots) slot.ensure(model.get(), arch, inputs.front().length, limit, selected != nullptr);
        monitor.stage("context_created");
        out << "],\"pooling_type\":" << int(llama_pooling_type(slots.front().ctx.get())) << ",\"rounds\":[";
        bool first = true;
        {
            Team team(slots, width);
            for (int phase = 0; phase < 2; ++phase) {
                const int count = phase == 0 ? o.warmup : o.iterations;
                for (int iteration = 0; iteration < count; ++iteration) for (size_t i = 0; i < inputs.size(); ++i) {
                    bool resized = false;
                    for (auto & slot : slots) {
                        resized |= slot.capacity < uint32_t(inputs[i].length);
                        slot.ensure(model.get(), arch, inputs[i].length, limit, selected != nullptr);
                    }
                    if (resized) monitor.stage("context_resized_" + std::to_string(inputs[i].length));
                    const double wall = team.run(inputs[i]);
                    if (!std::isfinite(wall) || wall <= 0) throw std::runtime_error("invalid wall time");
                    monitor.stage(std::string(phase == 0 ? "warmup" : "measured") + "_round_" + std::to_string(iteration) + "_input_" + std::to_string(i));
                    if (!first) out << ',';
                    first = false;
                    out << "{\"warmup\":" << (phase == 0 ? "true" : "false") << ",\"iteration\":" << iteration
                        << ",\"input_index\":" << i << ",\"tokens_per_job\":" << inputs[i].length << ",\"wall_ms\":" << wall
                        << ",\"jobs_per_second\":" << o.contexts * 1000.0 / wall
                        << ",\"tokens_per_second\":" << o.contexts * double(inputs[i].length) * 1000.0 / wall
                        << ",\"active_n_ctx\":" << llama_n_ctx(slots.front().ctx.get())
                        << ",\"active_n_batch\":" << llama_n_batch(slots.front().ctx.get())
                        << ",\"active_n_ubatch\":" << llama_n_ubatch(slots.front().ctx.get()) << ",\"jobs\":[";
                    for (size_t j = 0; j < team.jobs.size(); ++j) {
                        if (j) out << ',';
                        out << "{\"context_index\":" << j << ",\"latency_ms\":" << team.jobs[j].latency_ms << ",\"embedding\":[";
                        for (size_t d = 0; d < team.jobs[j].embedding.size(); ++d) { if (d) out << ','; out << team.jobs[j].embedding[d]; }
                        out << "]}";
                    }
                    out << "]}";
                    out.flush();
                    if (!out) throw std::runtime_error("output write failed");
                }
                if (phase == 0) monitor.stage("warmed");
            }
            monitor.stage("finished");
        }
        slots.clear();
        monitor.stage("contexts_freed");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        monitor.stage("contexts_freed_settled");
        out << "],\"memory\":";
        monitor.write(out);
        out << "}\n";
        out.close();
        if (!out) throw std::runtime_error("output close failed");
        return 0;
    } catch (const std::exception & e) {
        std::cerr << "llama-logits-poc: " << e.what() << '\n';
        return 1;
    }
}
