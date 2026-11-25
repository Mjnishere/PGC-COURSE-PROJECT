#include <omp.h>
#include "cpu_pipeline.h"

// Runs the CPU pipeline with a given thread count; returns best-of-reps timings.
static void run(const Config& c, const std::vector<float>& rawC, const std::vector<float>& rawQ, int threads,
                Result& res, double& t_pre, double& t_search) {
    t_pre = t_search = 1e30;
    for (int r = 0; r < c.reps; r++) {
        std::vector<float> C = rawC, Qm = rawQ;
        Timer t;
        t.start();
        cpu_normalize(C.data(), c.N, c.D, threads);
        cpu_normalize(Qm.data(), c.Q, c.D, threads);
        double a = t.ms();
        t.start();
        cpu_search(Qm.data(), C.data(), c.Q, c.N, c.D, c.K, threads, res);
        double b = t.ms();
        t_pre = std::min(t_pre, a);
        t_search = std::min(t_search, b);
    }
}

int main(int argc, char** argv) {
    Config c = parse_args(argc, argv);
    std::vector<float> rawC, rawQ;
    gen_data(rawC, c.N, c.D, c.seed);
    gen_data(rawQ, c.Q, c.D, c.seed + 1);
    int nt = omp_get_max_threads();

    Result serial, par;
    double p1, s1, pn, sn;
    run(c, rawC, rawQ, 1, serial, p1, s1);
    run(c, rawC, rawQ, nt, par, pn, sn);
    double recall, maxdiff;
    verify(serial, par, c.Q, c.K, recall, maxdiff);

    if (c.csv) {
        printf("impl,threads,N,D,Q,K,preprocess_ms,search_ms,total_ms\n");
        printf("cpu_serial,1,%d,%d,%d,%d,%.3f,%.3f,%.3f\n", c.N, c.D, c.Q, c.K, p1, s1, p1 + s1);
        printf("cpu_omp,%d,%d,%d,%d,%d,%.3f,%.3f,%.3f\n", nt, c.N, c.D, c.Q, c.K, pn, sn, pn + sn);
    } else {
        printf("N=%d D=%d Q=%d K=%d\n", c.N, c.D, c.Q, c.K);
        printf("%-12s %8s %14s %14s %12s\n", "impl", "threads", "preprocess ms", "score+topk ms", "total ms");
        printf("%-12s %8d %14.2f %14.2f %12.2f\n", "cpu_serial", 1, p1, s1, p1 + s1);
        printf("%-12s %8d %14.2f %14.2f %12.2f  (speedup %.2fx)\n", "cpu_omp", nt, pn, sn, pn + sn,
               (p1 + s1) / (pn + sn));
        printf("serial vs omp: recall@K=%.4f max score diff=%.2e\n", recall, maxdiff);
    }
    return 0;
}
