//
// Created by DanielSun on 2/12/2022.
//

#ifndef SEANN_ERRORHANDLER_CUH
#define SEANN_ERRORHANDLER_CUH

namespace seblas {
    class ErrorHandler {
    public:
         static void checkDeviceStatus(const char* file, int line) ;
    };
}


#endif //SEANN_ERRORHANDLER_CUH
