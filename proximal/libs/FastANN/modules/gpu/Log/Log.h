/*
 * Log.h
 *
 *  Created on: Jan 21, 2013
 *      Author: ytsai
 */

#ifndef LOG_H_
#define LOG_H_

#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <sys/time.h>

inline std::string NowTime() {
	char buffer[11];
	time_t t;
	time(&t);
	tm r = { 0 };
	strftime(buffer, sizeof(buffer), "%X", localtime_r(&t, &r));
	struct timeval tv;
	::gettimeofday(&tv, 0);
	char result[100] = { 0 };
	sprintf(result, "%s.%03ld", buffer, (long) tv.tv_usec / 1000);
	return result;
}

enum LogLevel {
	NVLOG_ERROR, NVLOG_WARNING, NVLOG_INFO, NVLOG_DEBUG
};

class Log {
public:
	Log();
	virtual ~Log();

	std::ostringstream& Get(LogLevel level = NVLOG_INFO);
	static LogLevel& ReportingLevel();
private:
	// no copy
	Log(const Log&);
	Log& operator =(const Log&);

	std::string ToString(LogLevel level) const;
private:
	std::ostringstream os_;
};

inline Log::Log() {
}

inline Log::~Log() {
	os_ << std::endl;
	fprintf(stderr, "%s", os_.str().c_str());
	fflush(stderr);
}

inline std::ostringstream& Log::Get(LogLevel level) {
	os_ << "- " << NowTime();
	os_ << " " << ToString(level) << ":\t";
	return os_;
}

inline LogLevel& Log::ReportingLevel() {
	static LogLevel reportingLevel = NVLOG_DEBUG;
	return reportingLevel;
}

inline std::string Log::ToString(LogLevel level) const {
	static const char* const buffer[] = { "ERROR", "WARNING", "INFO", "DEBUG" };
	return buffer[level];
}

#define NVLOG(level) \
if (level > Log::ReportingLevel()) ; \
else Log().Get(level)

#endif /* LOG_H_ */
