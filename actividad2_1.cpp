// ------------------------------------------------------------
// Actividad 2_1
// Tareas incluidas:
//   - Tarea 4: BFS para el costo (número mínimo de movimientos) hacia el estado objetivo ABCDEFGHIJKLMNO#.
//   - Tarea 5: A* con h1 (fichas fuera de lugar) -> imprimir costo mínimo.
//   - Tarea 6: A* con h2 (Manhattan) -> imprimir costo mínimo.
//   - Tarea 7: A* con h2 (Manhattan) -> imprimir la secuencia óptima de acciones (una por línea).
//               En todos los casos, si no hay solución, imprimir "UNSOLVABLE".
// ------------------------------------------------------------
// Cómo compilar (recomendado para compatibilidad C++11):
//   g++ -O2 -std=gnu++11 -o actividad2_1 actividad2_1.cpp
//
// Cómo ejecutar:
//   Tarea 4 (BFS costo):       ./actividad2_4 4     < estado >
/*  Tarea 5 (A* h1 costo):     ./actividad2_4 5     < estado >
    Tarea 6 (A* h2 costo):     ./actividad2_4 6     < estado >
    Tarea 7 (A* h2 acciones):  ./actividad2_4 7     < estado >
    donde <estado> es una cadena de 16 símbolos (incluyendo '#'). Ej: #AGCEBFDIJKHMNOL
*/
// ------------------------------------------------------------
#include <bits/stdc++.h>
using namespace std;

static const string GOAL = "ABCDEFGHIJKLMNO#";

// ---------------- Comunes ----------------
static inline int inversions(const string& s) {
    string t; t.reserve(16);
    for (char ch : s) if (ch != '#') t.push_back(ch);
    int inv = 0;
    for (int i = 0; i < (int)t.size(); ++i)
        for (int j = i + 1; j < (int)t.size(); ++j)
            if (t[i] > t[j]) ++inv;
    return inv;
}

// Regla correcta para 4x4 (ancho par):
// Soluble <=> (inversions + fila_del_blanco_desde_abajo) % 2 == 1
static inline bool solvable(const string& s) {
    int inv = inversions(s);
    int blank = (int)s.find('#');
    int rowFromBottom = 4 - (blank / 4); // 1..4 (1 = última fila)
    return ((inv + rowFromBottom) % 2) == 1;
}

struct Move { int dr, dc; const char* name; };
static const Move MOVES[4] = {
    {-1, 0, "UP"},
    {+1, 0, "DOWN"},
    { 0,-1, "LEFT"},
    { 0,+1, "RIGHT"},
};

static inline void vecinos(const string& s, vector<pair<string,string> >& out) {
    out.clear();
    int pos = (int)s.find('#');
    int r = pos / 4, c = pos % 4;
    for (int k = 0; k < 4; ++k) {
        int nr = r + MOVES[k].dr;
        int nc = c + MOVES[k].dc;
        if (nr >= 0 && nr < 4 && nc >= 0 && nc < 4) {
            string t = s;
            int np = nr * 4 + nc;
            swap(t[pos], t[np]);
            out.push_back(make_pair(t, string(MOVES[k].name)));
        }
    }
}

// h1: fichas fuera de lugar (ignorar '#')
static inline int h1_misplaced(const string& s) {
    int c = 0;
    for (int i = 0; i < 16; ++i)
        if (s[i] != '#' && s[i] != GOAL[i]) ++c;
    return c;
}

// Precomputar posiciones objetivo para Manhattan
static int goal_r[256], goal_c[256];
static void init_goal_pos() {
    for (int i = 0; i < 256; ++i) goal_r[i] = goal_c[i] = -1;
    for (int i = 0; i < 16; ++i) {
        char ch = GOAL[i];
        goal_r[(unsigned char)ch] = i / 4;
        goal_c[(unsigned char)ch] = i % 4;
    }
}

// h2: Manhattan (ignorar '#')
static inline int h2_manhattan(const string& s) {
    int sum = 0;
    for (int i = 0; i < 16; ++i) {
        char ch = s[i];
        if (ch == '#') continue;
        int r = i / 4, c = i % 4;
        int gr = goal_r[(unsigned char)ch], gc = goal_c[(unsigned char)ch];
        sum += abs(r - gr) + abs(c - gc);
    }
    return sum;
}

// ------------- BFS (Tarea 4) -------------
static int bfs_cost(const string& start) {
    if (start == GOAL) return 0;
    if (!solvable(start)) return -1;

    unordered_set<string> vis;
    vis.reserve(160000);
    queue< pair<string,int> > q;
    q.push(make_pair(start, 0));
    vis.insert(start);

    vector<pair<string,string> > neigh; neigh.reserve(4);
    while (!q.empty()) {
        pair<string,int> cur = q.front(); q.pop();
        const string& u = cur.first;
        int d = cur.second;

        vecinos(u, neigh);
        for (size_t i = 0; i < neigh.size(); ++i) {
            const string& v = neigh[i].first;
            if (vis.insert(v).second) {
                if (v == GOAL) return d + 1;
                q.push(make_pair(v, d + 1));
            }
        }
    }
    return -1;
}

// ------------- A* (Tareas 5,6,7) -------------
struct Node {
    int f, g;
    string state;
    // priority_queue en C++ es max-heap; invertimos para tener min-heap
    bool operator<(const Node& other) const {
        if (f != other.f) return f > other.f;
        // tie-break por g mayor (más profundo) para favorecer expansiones más cercanas a la meta
        return g < other.g;
    }
};

// A*: heur_id = 1 -> h1, heur_id = 2 -> h2
// Si path_out != NULL, se reconstruyen acciones
static int astar(const string& start, int heur_id, vector<string>* path_out) {
    if (start == GOAL) {
        if (path_out) path_out->clear();
        return 0;
    }
    if (!solvable(start)) return -1;

    // Estructuras A*
    priority_queue<Node> open;
    unordered_map<string,int> gcost; // mejor g conocido
    unordered_map<string, pair<string,string> > parent; // estado -> (padre, acción)

    gcost.reserve(200000);
    parent.reserve(200000);

    int h0 = (heur_id == 1 ? h1_misplaced(start) : h2_manhattan(start));
    open.push(Node{h0, 0, start});
    gcost[start] = 0;

    vector<pair<string,string> > neigh; neigh.reserve(4);

    while (!open.empty()) {
        Node cur = open.top(); open.pop();
        const string& u = cur.state;
        int g = cur.g;

        // Podría estar obsoleto si ya conocemos un g mejor
        unordered_map<string,int>::iterator itg = gcost.find(u);
        if (itg == gcost.end() || g > itg->second) continue;

        if (u == GOAL) {
            if (path_out) {
                // reconstruir
                vector<string> rev;
                string x = u;
                while (x != start) {
                    pair<string,string> pp = parent[x];
                    rev.push_back(pp.second);
                    x = pp.first;
                }
                reverse(rev.begin(), rev.end());
                *path_out = rev;
            }
            return g;
        }

        vecinos(u, neigh);
        for (size_t i = 0; i < neigh.size(); ++i) {
            const string& v = neigh[i].first;
            const string& act = neigh[i].second;
            int ng = g + 1;
            if (!gcost.count(v) || ng < gcost[v]) {
                gcost[v] = ng;
                int h = (heur_id == 1 ? h1_misplaced(v) : h2_manhattan(v));
                open.push(Node{ng + h, ng, v});
                if (path_out) parent[v] = make_pair(u, act);
            }
        }
    }
    return -1;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) {
        cerr << "Uso: ./actividad2_4 <tarea>\n"
                "  4 -> BFS costo\n"
                "  5 -> A* h1 (costo)\n"
                "  6 -> A* h2 (costo)\n"
                "  7 -> A* h2 (acciones)\n";
        return 1;
    }
    int tarea = atoi(argv[1]);
    if (tarea < 4 || tarea > 7) {
        cerr << "Tarea no válida. Use 4, 5, 6 o 7.\n";
        return 1;
    }

    string start;
    if (!(cin >> start) || start.size() != 16) return 0;

    // init datos para Manhattan
    init_goal_pos();

    if (tarea == 4) {
        int cost = bfs_cost(start);
        if (cost < 0) cout << "UNSOLVABLE\n";
        else cout << cost << "\n";
        return 0;
    }

    if (tarea == 5) {
        int cost = astar(start, 1, NULL);
        if (cost < 0) cout << "UNSOLVABLE\n";
        else cout << cost << "\n";
        return 0;
    }

    if (tarea == 6) {
        int cost = astar(start, 2, NULL);
        if (cost < 0) cout << "UNSOLVABLE\n";
        else cout << cost << "\n";
        return 0;
    }

    // tarea == 7
    vector<string> path;
    int cost = astar(start, 2, &path);
    if (cost < 0) {
        cout << "UNSOLVABLE\n";
        return 0;
    }
    // Imprimir acciones (una por línea). Si se desea, antes se puede imprimir el costo.
    // cout << cost << "\n";
    for (size_t i = 0; i < path.size(); ++i) cout << path[i] << "\n";
    return 0;
}