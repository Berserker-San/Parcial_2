// ------------------------------------------------------------
// Actividad 2
// Tareas incluidas:
//   - Tarea 3: Listar acciones válidas desde un estado en el orden: UP, DOWN, LEFT, RIGHT.
// ------------------------------------------------------------
// Cómo compilar:
//     g++ -O2 -std=c++17 -o actividad2 actividad2.cpp
//
// Cómo ejecutar (Tarea 3):
//     ./actividad2 3 => una línea con la cadena de 16 símbolos.
// ------------------------------------------------------------

#include <bits/stdc++.h>
using namespace std;

int main(int argc, char** argv) {
    ios::sync_with_stdio(false);
    cin.tie(nullptr);

    if (argc < 2) {
        cerr << "Uso: ./actividad2 3\n";
        return 1;
    }
    int tarea = atoi(argv[1]);
    if (tarea != 3) {
        cerr << "Tarea no válida. Solo Tarea 3 en este archivo.\n";
        return 1;
    }

    string s;
    if (!(cin >> s) || s.size() != 16) return 0;
    int pos = (int)s.find('#');
    int r = pos / 4, c = pos % 4;

    if (r > 0) cout << "UP\n";
    if (r < 3) cout << "DOWN\n";
    if (c > 0) cout << "LEFT\n";
    if (c < 3) cout << "RIGHT\n";
    return 0;
}
