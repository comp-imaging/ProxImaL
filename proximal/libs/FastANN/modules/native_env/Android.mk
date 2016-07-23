LOCAL_PATH := $(call my-dir)

# module definition
include $(CLEAR_VARS)
LOCAL_MODULE := native_env

LOCAL_CFLAGS := -Wall -Wcast-align -Wextra -std=gnu++0x -DARCH_ARM -DDEBUG_MODE

ifeq ($(NDK_DEBUG),1)
  LOCAL_CFLAGS += -DDEBUG
else
  LOCAL_CFLAGS += -O3 -mtune=cortex-a15 -march=armv7-a -ffast-math
endif

LOCAL_ARM_NEON     := true
ARCH_ARM_HAVE_NEON := true

SRC_PATH   := $(LOCAL_PATH)
MY_SOURCES := $(wildcard $(SRC_PATH)/*.cpp)
MY_SOURCES += $(wildcard $(SRC_PATH)/ip/*.cpp)

LOCAL_SRC_FILES         := $(MY_SOURCES:$(LOCAL_PATH)/%=%)
LOCAL_C_INCLUDES        := $(SRC_PATH)
LOCAL_STATIC_LIBRARIES  := libnative_env_core
LOCAL_EXPORT_C_INCLUDES := $(SRC_PATH)
LOCAL_EXPORT_LDLIBS     := -landroid

# STL support
#LOCAL_C_INCLUDES    += $(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.6/include
#LOCAL_C_INCLUDES    += $(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.6/libs/armeabi-v7a/include
#LOCAL_EXPORT_LDLIBS += $(NDK_ROOT)/sources/cxx-stl/gnu-libstdc++/4.6/libs/armeabi-v7a/libgnustl_static.a

include $(BUILD_STATIC_LIBRARY)

# Import shared modules
$(call import-module,native_env_core)
