#include <bits/stdc++.h>
#include <omp.h>
using namespace std;

static const string GOAL = "ABCDEFGHIJKLMNO#";

inline vector<string> vecinos(const string &s)
{
    int pos = s.find('#');
    int r = pos / 4, c = pos % 4;
    vector<string> nb;
    auto swp = [&](int np)
    {
        string t = s;
        swap(t[pos], t[np]);
        nb.push_back(t);
    };
    if (c < 3)
        swp(pos + 1);
    if (c > 0)
        swp(pos - 1);
    if (r > 0)
        swp(pos - 4);
    if (r < 3)
        swp(pos + 4);
    return nb;
}

struct Result
{
    int depth;
    double seconds;
    uint64_t expanded;
};

Result bfs_openmp(const string &start, int P)
{
    double t0 = omp_get_wtime();

    if (start == GOAL)
        return {0, omp_get_wtime() - t0, 0};

    unordered_set<string> visited;
    visited.reserve(1 << 20);
    vector<string> frontier{start};
    visited.insert(start);

    int depth = 0;
    uint64_t expanded_total = 0;
    bool found = false;

    while (!frontier.empty() && !found)
    {
        vector<string> next_layer;
#pragma omp parallel num_threads(P)
        {
            vector<string> local_next;
            uint64_t local_exp = 0;

#pragma omp for schedule(static)
            for (int i = 0; i < (int)frontier.size(); ++i)
            {
                if (found)
                    continue;
                const string &s = frontier[i];
                local_exp++;
                auto nbs = vecinos(s);
                for (auto &ns : nbs)
                {
                    if (ns == GOAL)
                    {
                        found = true;
                        break;
                    }
                    local_next.push_back(ns);
                }
            }

#pragma omp critical
            {
                expanded_total += local_exp;
                next_layer.insert(next_layer.end(), local_next.begin(), local_next.end());
            }
        }

        // deduplicación y actualización del visitado global
        vector<string> unique_next;
        unique_next.reserve(next_layer.size());
        for (auto &ns : next_layer)
        {
            if (!visited.count(ns))
            {
                visited.insert(ns);
                unique_next.push_back(ns);
            }
        }

        frontier.swap(unique_next);
        depth++;
    }

    double t1 = omp_get_wtime();
    return {found ? depth : -1, t1 - t0, expanded_total};
}

int main()
{
    string input;
    cout << "Ingrese cadena de 16 caracteres (# como espacio): ";
    cin >> input;

    vector<int> threads = {1, 2, 4};
    cout << "\nResultados OpenMP BFS:\n";
    cout << "Threads,Tiempo(s),Speedup,Eficiencia,Expandidos\n";

    Result seq = bfs_openmp(input, 1);
    for (int P : threads)
    {
        Result r = bfs_openmp(input, P);
        double speedup = seq.seconds / r.seconds;
        double eff = speedup / P;
        cout << P << "," << r.seconds << "," << speedup << "," << eff << "," << r.expanded << "\n";
    }
}

