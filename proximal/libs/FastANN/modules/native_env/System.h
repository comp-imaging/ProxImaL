/*
 * Copyright (c) 2012-2013, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA Corporation and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA Corporation is strictly prohibited.
 */

#ifndef SYSTEM_H_
#define SYSTEM_H_

#include "SystemCore.h"

// device key codes
#define SK_LEFT  0
#define SK_RIGHT 1
#define SK_DOWN  2
#define SK_UP    3
#define SK_FIRE  4
#define SK_LSOFT 5
#define SK_RSOFT 6
#define SK_HASH  7
#define SK_STAR  8
#define SK_CLEAR 9
#define SK_KEY0  10
#define SK_KEY1  11
#define SK_KEY2  12
#define SK_KEY3  13
#define SK_KEY4  14
#define SK_KEY5  15
#define SK_KEY6  16
#define SK_KEY7  17
#define SK_KEY8  18
#define SK_KEY9  19

#define SK_NONE  20

// extended full-keyboard codes
#define SK_EX_A 30
#define SK_EX_B 31
#define SK_EX_C 32
#define SK_EX_D 33
#define SK_EX_E 34
#define SK_EX_F 35
#define SK_EX_G 36
#define SK_EX_H 37
#define SK_EX_I 38
#define SK_EX_J 39
#define SK_EX_K 40
#define SK_EX_L 41
#define SK_EX_M 42
#define SK_EX_N 43
#define SK_EX_O 44
#define SK_EX_P 45
#define SK_EX_Q 46
#define SK_EX_R 47
#define SK_EX_S 48
#define SK_EX_T 49
#define SK_EX_U 50
#define SK_EX_V 51
#define SK_EX_W 52
#define SK_EX_X 53
#define SK_EX_Y 54
#define SK_EX_Z 55

#define SK_EX_SPACE     60
#define SK_EX_BKSPACE   61
#define SK_EX_ENTER     62
#define SK_EX_INSERT    63
#define SK_EX_DELETE    64
#define SK_EX_HOME      65
#define SK_EX_END       66
#define SK_EX_PAGEUP    67
#define SK_EX_PAGEDOWN  68
#define SK_EX_EQUALS    69
#define SK_EX_MINUS     70
#define SK_EX_SLASH     71
#define SK_EX_BKSLASH   72
#define SK_EX_DOT       73
#define SK_EX_COMMA     74
#define SK_EX_SEMICOLON 75
#define SK_EX_LBRACKET  76
#define SK_EX_RBRACKET  77
#define SK_EX_LSHIFT    78
#define SK_EX_RSHIFT    79
#define SK_EX_LCTRL     80
#define SK_EX_RCTRL     81
#define SK_EX_TILDE     82
#define SK_EX_TAB       83
#define SK_EX_LALT      84
#define SK_EX_RALT      85
#define SK_EX_ESCAPE    86
#define SK_EX_F1        87
#define SK_EX_F2        88
#define SK_EX_F3        89
#define SK_EX_F4        90
#define SK_EX_F5        91
#define SK_EX_F6        92
#define SK_EX_F7        93
#define SK_EX_F8        94
#define SK_EX_F9        95
#define SK_EX_F10       96
#define SK_EX_F11       97
#define SK_EX_F12       98

namespace System
{

// C-like global access state-less API

// USER INPUT
int KeyIsPressed(int key, int delay);
int KeyIsPressed(int key);
int KeyIsReleased(int key);
int KeyIsDown(int key);
int KeyGetLastPressedChar(void);
void KeyReset(int key);

// DISPLAY
uint ScreenGetWidth(void);
uint ScreenGetHeight(void);

// TIMER
double TimerGetTime(void);
double TimerGetFrameTime(void);

}

#endif /* SYSTEM_H_ */
