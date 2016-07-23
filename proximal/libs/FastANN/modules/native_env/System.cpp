/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#include "System.h"
#include "DeviceProxy.h"

/**
 * Checks whether a key was pressed or repeated.
 * @param key - key code (SK_*)
 * @param delay - repeat delay. The delay after a press is twice the value.
 * @return true(1) if the key is pressed, false(0) otherwise
 */
int System::KeyIsPressed(int key, int delay)
{
    return Internal::DeviceProxy::GetInstance()->keyIsPressed(key, delay);
}

/**
 * Checks whether a key was pressed
 * @param key - key code (SK_*)
 * @return true(1) if the key is pressed, false(0) otherwise
 */
int System::KeyIsPressed(int key)
{
    return Internal::DeviceProxy::GetInstance()->keyIsPressed(key);
}

/**
 * Checks whether a key was released
 * @param key - key code (SK_*)
 * @return true(1) if the key is released, false(0) otherwise
 */
int System::KeyIsReleased(int key)
{
    return Internal::DeviceProxy::GetInstance()->keyIsReleased(key);
}

/**
 * Checks whether a key is down
 * @param key - key code (SK_*)
 * @return true(1) if the key is down, false(0) otherwise
 */
int System::KeyIsDown(int key)
{
    return Internal::DeviceProxy::GetInstance()->keyIsDown(key);
}

/**
 * Resets key state
 * @param key - key code (SK_*)
 */
void System::KeyReset(int key)
{
    Internal::DeviceProxy::GetInstance()->keyReset(key);
}

/**
 * Extended keyboard support function. Returns the ASCII code of 
 * last pressed character
 * @return ASCII code of last pressed key
 */
int System::KeyGetLastPressedChar(void)
{
    return Internal::DeviceProxy::GetInstance()->keyGetLastPressedChar();
}

// ==========================================================================
// TIMER
// ==========================================================================

double System::TimerGetTime(void)
{
    return Internal::DeviceProxy::GetInstance()->getTime();
}

double System::TimerGetFrameTime(void)
{
    return Internal::DeviceProxy::GetInstance()->getFrameTime();
}

// ==========================================================================
// DISPLAY
// ==========================================================================

uint System::ScreenGetWidth(void)
{
    return Internal::DeviceProxy::GetInstance()->getScreenPixelWidth();
}

uint System::ScreenGetHeight(void)
{
    return Internal::DeviceProxy::GetInstance()->getScreenPixelHeight();
}

