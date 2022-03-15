//
// Created by DanielSun on 3/14/2022.
//

#ifndef SEANN_LOGUTILS_CUH
#define SEANN_LOGUTILS_CUH

#include <string>
#include "Color.cuh"
using namespace std;

namespace seio {

    enum LogLevel {
        LOG_LEVEL_DEBUG = 0,
        LOG_LEVEL_INFO = 1,
        LOG_LEVEL_WARN = 2,
        LOG_LEVEL_ERROR = 3,
        LOG_LEVEL_FATAL = 4
    };

    enum LogSegments {
        LOG_SEG_SEIO = 0,
        LOG_SEG_SEANN = 1,
        LOG_SEG_SEBLAS = 2,
    };

    void printLogHead(LogLevel level, LogSegments segment);

    void logInfo(LogSegments seg, string msg);

    void logDebug(LogSegments seg, string msg);

    void logWarn(LogSegments seg, string msg);

    void logError(LogSegments seg, string msg);

    void logFatal(LogSegments seg, string msg);
}


#endif //SEANN_LOGUTILS_CUH
