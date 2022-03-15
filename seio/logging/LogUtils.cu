//
// Created by DanielSun on 3/14/2022.
//

#include <utility>

#include "LogUtils.cuh"

namespace seio {
    void printLogHead(LogLevel level, LogSegments segment){

        dye::colorful<basic_string<char>> segName;
        switch(segment){
            case LogSegments::LOG_SEG_SEANN:
                segName = dye::bright_white("seann");
                break;
            case LogSegments::LOG_SEG_SEIO:
                segName = dye::light_purple("seio");
                break;
            case LogSegments::LOG_SEG_SEBLAS:
                segName = dye::light_yellow("seblas");
                break;
        }

        dye::colorful<basic_string<char>> levelPrefix;
        switch(level){
            case LogLevel::LOG_LEVEL_DEBUG:
                levelPrefix = dye::light_aqua("DEBUG");
                break;
            case LogLevel::LOG_LEVEL_INFO:
                levelPrefix = dye::light_blue("INFO");
                break;
            case LogLevel::LOG_LEVEL_WARN:
                levelPrefix = dye::light_yellow("WARN");
                break;
            case LogLevel::LOG_LEVEL_ERROR:
                levelPrefix = dye::red("ERROR");
                break;
            case LogLevel::LOG_LEVEL_FATAL:
                levelPrefix = dye::red("FATAL");
                break;
        }

        time_t secs = time(nullptr);
        struct tm *local = localtime(&secs);

        //print the current time
        cout<<dye::aqua("[")<<dye::light_red(local->tm_hour)<<dye::light_aqua(":")
        <<dye::light_red(local->tm_min)<<dye::light_aqua(":")<<dye::light_red(local->tm_sec);

        //print the log segment
        cout<<dye::aqua("|")<<segName<<dye::aqua("]");

        //print the log level
        if(level == LogLevel::LOG_LEVEL_ERROR || level == LogLevel::LOG_LEVEL_FATAL)
            cout<<dye::red(": ")<<levelPrefix<<dye::red(" >>> ");
        else
            cout<<dye::purple(": ")<<levelPrefix<<dye::purple(" >>> ");
    }

    void logInfo(LogSegments segment, string msg){
        printLogHead(LogLevel::LOG_LEVEL_INFO, segment);
        cout<<dye::blue(std::move(msg))<<endl;
    }

    void logDebug(LogSegments seg, string msg){
        printLogHead(LogLevel::LOG_LEVEL_DEBUG, seg);
        cout<<dye::aqua(std::move(msg))<<endl;
    }

    void logWarn(LogSegments seg, string msg){
        printLogHead(LogLevel::LOG_LEVEL_WARN, seg);
        cout<<dye::yellow(std::move(msg))<<endl;
    }

    void logError(LogSegments seg, string msg){
        printLogHead(LogLevel::LOG_LEVEL_ERROR, seg);
        cout<<dye::red(std::move(msg))<<endl;
    }

    void logFatal(LogSegments seg, string msg){
        printLogHead(LogLevel::LOG_LEVEL_FATAL, seg);
        cout<<dye::red(std::move(msg))<<endl;
    }
}