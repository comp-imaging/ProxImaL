/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef DEVICEPROXY_H_
#define DEVICEPROXY_H_

#include <vector>
#include <string>
#include "Base.h"
#include "SystemTimer.h"
#include "ManagedPtr.h"

#ifdef ANDROID
#include <jni.h>
struct AAssetManager;
#endif

namespace System
{

namespace Internal
{

// internal system object

class DeviceProxy: public ManagedAbstractObject
{
public:
    enum Components
    {
        ComponentProxyInstance = 0, ComponentCamera = 1, ComponentFileCacheBlockAllocator = 2
    };

    enum
    {
        TotalKeyCount = 128, TotalComponentCount = 3
    };

    DeviceProxy(void);
    ~DeviceProxy(void);

    uint getScreenPixelWidth(void);
    void setScreenPixelWidth(uint value);
    uint getScreenPixelHeight(void);
    void setScreenPixelHeight(uint value);

    void updateKeyboardState(double time);
    bool keyIsPressed(int key, int delay);
    bool keyIsPressed(int key);
    bool keyIsReleased(int key);
    bool keyIsDown(int key);
    int keyGetLastPressedChar(void);
    void keySetLastPressedChar(int key);
    void keyReset(int key);

    void dispatchKeyDown(int key);
    void dispatchKeyUp(int key);

    double getTime(void);
    double getFrameTime(void);

    std::string const &getExternalPath(void);
    void setExternalPath(std::string const &path);

#ifdef ANDROID
    AAssetManager *getNativeAssetManager(void);
    void setNativeAssetManager(AAssetManager *instance);

    JNIEnv *getNativeJNIEnvironment(void);
    void setNativeJNIEnvironment(JNIEnv *instance);

    jobject getNativeClassInstance(void);
    void setNativeClassInstance(jobject instance);
#endif

    static std::vector<managed_ptr<ManagedAbstractObject> > &GetComponents(void);
    static DeviceProxy *GetInstance(void);

private:
    Timer mTimer;

    double mLastKeyEventTime;
    double mInterruptTime;
    double mInterruptShift;

    // user input support
    int mKeyLastPressedChar;
    uint mKeyState[TotalKeyCount >> 5], mKeyPrevState[TotalKeyCount >> 5];
    uint mKeyPressed[TotalKeyCount >> 5], mKeyReleased[TotalKeyCount >> 5];
    double mKeyPressTime[TotalKeyCount], mKeyLastRepeatTime[TotalKeyCount];

    std::string mExternalPath;
    uint mScreenWidth;
    uint mScreenHeight;

#ifdef ANDROID
    AAssetManager *mNativeAssetManager;
    JNIEnv *mNativeJNIEnvironment;
    jobject mNativeClassInstance;
#endif
};

}

}

#endif /* DEVICEPROXY_H_ */
