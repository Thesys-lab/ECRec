#ifndef PS_PLUS_SERVER_MLC_UTILS_H_
#define PS_PLUS_SERVER_MLC_UTILS_H_

// #include <string>
// #include <vector>
// #include <stdlib.h>
// #include <unordered_set>
#include <atomic>

// namespace ps {
// namespace server {
static int WRITE_NUM = 3;
static int INTERVAL = 4;

std::atomic<int> v(0);

// static int interval = 3;

bool getNext(int writeNum, int interval) {
    // cout << "v is " << v << endl;
    v = v%interval;
    v++;
    return v <= writeNum;
}

// }
// }

#endif // PS_PLUS_SERVER_MLC_UTILS_H_
