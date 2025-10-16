// ------------------------------------------------------------
// Actividad 1
// Tareas incluidas:
//   - Tarea 1: Imprimir tablero 4x4
//   - Tarea 2: Aplicar un movimiento
// ------------------------------------------------------------
// Cómo compilar: g++ -O2 -std=c++17 -o actividad1 actividad1.cpp
//
// Cómo ejecutar:
//   - Tarea 1 (imprimir tablero):
//       ./actividad1 1 => una línea con la cadena de 16 símbolos (p.ej. ELFIGHJONAKDMB#C)
//   - Tarea 2 (aplicar movimiento):
//       ./actividad1 2 => primera línea: cadena de 16 símbolos
//                         segunda línea: UP | DOWN | LEFT | RIGHT
// ------------------------------------------------------------

#include <bits/stdc++.h>
using namespace std;

static inline void imprimeTablero(const string& s) {
    for (int i = 0; i < 16; ++i) {
        cout << s[i] << (i % 4 == 3 ? '\n' : ' ');
    }
}

static inline string aplicaMovimiento(const string& s, const string& cmd) {
    int pos = (int)s.find('#');
    int r = pos / 4, c = pos % 4;
    int nr = r, nc = c;
    if (cmd == "UP" && r > 0)            --nr;
    else if (cmd == "DOWN" && r < 3)     ++nr;
    else if (cmd == "LEFT" && c > 0)     --nc;
    else if (cmd == "RIGHT" && c < 3)    ++nc;
    else return s; // acción inválida (manteniene el estado)

    string t = s;
    swap(t[pos], t[nr * 4 + nc]);
    return t;
}

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) {
        cerr << "Uso: ./actividad1 <tarea>\n"
             << "  tarea = 1  -> Imprimir tablero 4x4\n"
             << "  tarea = 2  -> Aplicar movimiento y mostrar tablero\n";
        return 1;
    }

    int tarea = atoi(argv[1]);

    if (tarea == 1) {
        string s;
        if (!(cin >> s) || s.size() != 16) return 0;
        imprimeTablero(s);
    } else if (tarea == 2) {
        string s, cmd;
        if (!(cin >> s) || s.size() != 16) return 0;
        if (!(cin >> cmd)) return 0;
        string t = aplicaMovimiento(s, cmd);
        imprimeTablero(t);
    } else {
        cerr << "Tarea no válida. Use 1 o 2.\n";
        return 1;
    }
    return 0;
}