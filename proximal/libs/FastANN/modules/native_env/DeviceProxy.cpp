/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifdef ANDROID
#include <android/log.h>
#define ANDROID_MODULE_NAME "nv_rcf"
#endif
#include <stdio.h>
#ifdef LINUX
#include <wchar.h>
#endif
#include "DeviceProxy.h"
#include "System.h"

#define SK_ISSET(ptr,i) (((ptr)[(i)>>5]&(1<<((i)&0x1f)))!=0)
#define SK_SET(ptr,i)   ((ptr)[(i)>>5]|=1<<((i)&0x1f))
#define SK_RESET(ptr,i) ((ptr)[(i)>>5]&=~(1<<((i)&0x1f)))

static std::vector<managed_ptr<ManagedAbstractObject> > gDeviceProxyComponents(
        System::Internal::DeviceProxy::TotalComponentCount);

std::vector<managed_ptr<ManagedAbstractObject> > &System::Internal::DeviceProxy::GetComponents(void)
{
    return gDeviceProxyComponents;
}

System::Internal::DeviceProxy *System::Internal::DeviceProxy::GetInstance(void)
{
    std::vector<managed_ptr<ManagedAbstractObject> > &components(GetComponents());

    DeviceProxy *instance = static_cast<DeviceProxy *>(components[ComponentProxyInstance].get());
    if (instance == 0)
    {
        instance = new DeviceProxy();
        components[ComponentProxyInstance] = instance;
    }

    return instance;
}

// ==========================================================================
// INPUT
// ==========================================================================

/**
 * Performs initialization of global, system independent data. If initialization fails application
 * should not run.
 * @return true if initialization was a success, false otherwise
 */
System::Internal::DeviceProxy::DeviceProxy()
        : mTimer(), mLastKeyEventTime(0), mInterruptTime(0), mInterruptShift(0), mKeyLastPressedChar(0),
          mExternalPath(), mScreenWidth(0), mScreenHeight(0)
{
    // user input support
    System::MemorySet(mKeyState, 0, sizeof(uint) * (TotalKeyCount >> 5));
    System::MemorySet(mKeyPrevState, 0, sizeof(uint) * (TotalKeyCount >> 5));
    System::MemorySet(mKeyPressed, 0, sizeof(uint) * (TotalKeyCount >> 5));
    System::MemorySet(mKeyReleased, 0, sizeof(uint) * (TotalKeyCount >> 5));
    System::MemorySet(mKeyPressTime, 0, sizeof(double) * TotalKeyCount);
    System::MemorySet(mKeyLastRepeatTime, 0, sizeof(double) * TotalKeyCount);

#ifdef ANDROID
    mNativeAssetManager = 0;
    mNativeJNIEnvironment = 0;
    mNativeClassInstance = 0;
#endif
}

System::Internal::DeviceProxy::~DeviceProxy()
{
}

uint System::Internal::DeviceProxy::getScreenPixelWidth(void)
{
    return mScreenWidth;
}

uint System::Internal::DeviceProxy::getScreenPixelHeight(void)
{
    return mScreenHeight;
}

void System::Internal::DeviceProxy::setScreenPixelWidth(uint value)
{
    mScreenWidth = value;
}

void System::Internal::DeviceProxy::setScreenPixelHeight(uint value)
{
    mScreenHeight = value;
}

/**
 * Updates keyboard state variables and global frame timer
 */
void System::Internal::DeviceProxy::updateKeyboardState(double time)
{
    mLastKeyEventTime = time;

    for (int i = 0; i < (TotalKeyCount >> 5); i++)
    {
        mKeyPressed[i] = mKeyState[i] & (mKeyState[i] ^ mKeyPrevState[i]);
        mKeyReleased[i] = mKeyPrevState[i] & (mKeyPrevState[i] ^ mKeyState[i]);
        mKeyPrevState[i] = mKeyState[i];
    }
}

bool System::Internal::DeviceProxy::keyIsPressed(int key, int delay)
{
    if (!SK_ISSET( mKeyState, key ))
    {
        return 0;
    }

    double repeatTime = mKeyLastRepeatTime[key];
    if (repeatTime != 0 && repeatTime > mKeyPressTime[key] + delay)
    {
        delay >>= 1;
    }

    if (repeatTime + delay > mLastKeyEventTime)
    {
        return 0;
    }

    mKeyLastRepeatTime[key] = mLastKeyEventTime;

    return 1;
}

bool System::Internal::DeviceProxy::keyIsPressed(int key)
{
    return SK_ISSET( mKeyPressed, key );
}

/**
 * Checks whether a key was released
 * @param key - key code (SK_*)
 * @return true(1) if the key is released, false(0) otherwise
 */
bool System::Internal::DeviceProxy::keyIsReleased(int key)
{
    return SK_ISSET( mKeyReleased, key );
}

/**
 * Checks whether a key is down
 * @param key - key code (SK_*)
 * @return true(1) if the key is down, false(0) otherwise
 */
bool System::Internal::DeviceProxy::keyIsDown(int key)
{
    return SK_ISSET( mKeyState, key );
}

/**
 * Extended keyboard support function. Returns the ASCII code of
 * last pressed character
 * @return ASCII code of last pressed key
 */
int System::Internal::DeviceProxy::keyGetLastPressedChar(void)
{
    return mKeyLastPressedChar;
}

/**
 * Extended keyboard support function. Sets the ASCII code of
 * last pressed character
 * @return ASCII code of last pressed key
 */
void System::Internal::DeviceProxy::keySetLastPressedChar(int key)
{
    mKeyLastPressedChar = key;
}

/**
 * Resets key state
 * @param key - key code (SK_*)
 */
void System::Internal::DeviceProxy::keyReset(int key)
{
    SK_RESET( mKeyState, key);
    SK_RESET( mKeyPressed, key);
    SK_RESET( mKeyReleased, key);
}

void System::Internal::DeviceProxy::dispatchKeyDown(int key)
{
    if (!SK_ISSET( mKeyState, key ))
    {
        SK_SET( mKeyState, key);
        mKeyPressTime[key] = mLastKeyEventTime;
        mKeyLastRepeatTime[key] = 0;
    }
}

void System::Internal::DeviceProxy::dispatchKeyUp(int key)
{
    SK_RESET( mKeyState, key);
}

double System::Internal::DeviceProxy::getTime(void)
{
    return mTimer.get();
}

double System::Internal::DeviceProxy::getFrameTime(void)
{
    return mLastKeyEventTime;
}

std::string const &System::Internal::DeviceProxy::getExternalPath(void)
{
    return mExternalPath;
}

void System::Internal::DeviceProxy::setExternalPath(std::string const &path)
{
    mExternalPath = path;
}

#ifdef ANDROID

AAssetManager *System::Internal::DeviceProxy::getNativeAssetManager(void)
{
    return mNativeAssetManager;
}

void System::Internal::DeviceProxy::setNativeAssetManager(AAssetManager *instance)
{
    mNativeAssetManager = instance;
}

JNIEnv *System::Internal::DeviceProxy::getNativeJNIEnvironment(void)
{
    return mNativeJNIEnvironment;
}

void System::Internal::DeviceProxy::setNativeJNIEnvironment(JNIEnv *instance)
{
    mNativeJNIEnvironment = instance;
}

jobject System::Internal::DeviceProxy::getNativeClassInstance(void)
{
    return mNativeClassInstance;
}

void System::Internal::DeviceProxy::setNativeClassInstance(jobject instance)
{
    mNativeClassInstance = instance;
}

#endif
