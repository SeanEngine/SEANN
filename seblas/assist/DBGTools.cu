//
// Created by DanielSun on 2/15/2022.
//

#include "DBGTools.cuh"
#include "iostream"

using namespace std;
void seblas::inspect(Tensor *A) {
    auto* debug = Tensor::declare(A->dims)->createHost();
    copyD2H(A, debug);

    for (int w0 = 0; w0 < debug->dims.n; w0++) {
        for (int depth0 = 0; depth0 < debug->dims.c; depth0++) {
            for (int row0 = 0; row0 < debug->dims.rows; row0++) {
                for (int col0 = 0; col0 < debug->dims.cols; col0++) {
                    cout << debug->get(w0, depth0, row0, col0) << " ";
                }
                cout << endl;
            }
            cout << "\n" << endl;
        }
        cout << "\n\n" << endl;
    }



    Tensor::destroyHost(debug);
}