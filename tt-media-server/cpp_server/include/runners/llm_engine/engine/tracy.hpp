#pragma once

#if defined(LLM_ENGINE_TRACY)
#include <tracy/Tracy.hpp>
#include <common/TracySystem.hpp>
#define TracySetThreadName(name) ::tracy::SetThreadName(name)
#else
#define ZoneScoped
#define ZoneScopedN(x)
#define ZoneNamedN(varname, name, active)
#define ZoneTransientN(varname, name, active)
#define ZoneTextF(fmt, ...)
#define ZoneTextVF(varname, fmt, ...)
#define FrameMark
#define TracySetThreadName(name)
#define TracySetProgramName(name)
#endif
