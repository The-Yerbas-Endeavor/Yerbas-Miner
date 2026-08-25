// Persistent per-device CN-Fast setup/coop/final geometry cache.
// Cache keys include GPU identity, driver/runtime and active production batch,
// so a winner learned on one card is never blindly reused on another.

namespace {

constexpr int kCnFastMicroCacheRevision = 1;

std::filesystem::path cn_fast_micro_cache_path(int device_id,
                                               const cudaDeviceProp& props,
                                               std::size_t batch_size)
{
    int driver_version = 0;
    int runtime_version = 0;
    if (cudaDriverGetVersion(&driver_version) != cudaSuccess ||
        cudaRuntimeGetVersion(&runtime_version) != cudaSuccess) {
        cudaGetLastError();
        return {};
    }
    const std::string key = cn_cache_key(device_id, props, batch_size,
                                         driver_version, runtime_version);
    return cn_cache_directory() / ("cn-fast-micro-rev" +
        std::to_string(kCnFastMicroCacheRevision) + '-' + key + ".txt");
}

bool valid_cn_fast_micro_threads(int value, const cudaDeviceProp& props)
{
    return value >= 32 && value <= props.maxThreadsPerBlock &&
           (value == 32 || value == 64 || value == 128 || value == 256);
}

bool load_cn_fast_micro_cache(int device_id,
                              const cudaDeviceProp& props,
                              std::size_t batch_size,
                              CnGeometry& g)
{
    const char* force = std::getenv("YERBAS_CUDA_CN_FAST_BENCHMARK");
    if (force && *force && std::string(force) != "0") return false;
    const char* retune = std::getenv("YERBAS_CUDA_RETUNE");
    if (retune && *retune && std::string(retune) != "0") return false;

    const auto path = cn_fast_micro_cache_path(device_id, props, batch_size);
    if (path.empty()) return false;
    std::ifstream in(path);
    if (!in) return false;

    std::string magic;
    int revision = 0;
    int setup_threads = 0;
    int coop_threads = 0;
    int final_threads = 0;
    if (!(in >> magic >> revision >> setup_threads >> coop_threads >> final_threads) ||
        magic != "YERBAS_CN_FAST_MICRO" || revision != kCnFastMicroCacheRevision ||
        !valid_cn_fast_micro_threads(setup_threads, props) ||
        !valid_cn_fast_micro_threads(coop_threads, props) ||
        !valid_cn_fast_micro_threads(final_threads, props))
        return false;

    g.setup_threads = setup_threads;
    g.final_threads = final_threads;
    if (device_id >= 0 && device_id < kMaxLoopTuneDevices)
        g_cn_coop_threads[device_id][2] = coop_threads;

    std::cout << "[GPU " << device_id << "] CN-Fast micro cache"
              << " | setup=" << setup_threads
              << " | coop=" << coop_threads
              << " | final=" << final_threads << '\n';
    return true;
}

void save_cn_fast_micro_cache(int device_id,
                              const cudaDeviceProp& props,
                              std::size_t batch_size,
                              int setup_threads,
                              int coop_threads,
                              int final_threads)
{
    try {
        const auto path = cn_fast_micro_cache_path(device_id, props, batch_size);
        if (path.empty()) return;
        std::filesystem::create_directories(path.parent_path());
        const auto temp = path.string() + ".tmp";
        std::ofstream out(temp, std::ios::trunc);
        if (!out) return;
        out << "YERBAS_CN_FAST_MICRO " << kCnFastMicroCacheRevision << ' '
            << setup_threads << ' ' << coop_threads << ' ' << final_threads << '\n';
        out.close();
        if (!out) return;
        std::error_code ec;
        std::filesystem::rename(temp, path, ec);
        if (ec) {
            std::filesystem::remove(path, ec);
            ec.clear();
            std::filesystem::rename(temp, path, ec);
        }
    } catch (...) {
        // Cache failures are non-fatal; the safe production geometry remains active.
    }
}

} // anonymous namespace
