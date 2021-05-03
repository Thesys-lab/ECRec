#ifndef PS_PLUS_SERVER_MLC_UTILS_H_
#define PS_PLUS_SERVER_MLC_UTILS_H_

// #include <string>
// #include <vector>
// #include <stdlib.h>
// #include <unordered_set>
#include <atomic>

// namespace ps {
// namespace server {
static int INTERVAL = 4;

std::atomic<int> v(0);

// static int interval = 3;

bool getNext(int interval) {
    // cout << "v is " << v << endl;
    if (v % interval == 0) {
        v = 1;
        return true;
    }
    v++;
    return false;
}

// }
// }

#endif // PS_PLUS_SERVER_MLC_UTILS_H_