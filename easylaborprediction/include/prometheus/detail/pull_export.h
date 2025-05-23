
#ifndef PROMETHEUS_CPP_PULL_EXPORT_H
#define PROMETHEUS_CPP_PULL_EXPORT_H

#ifdef PROMETHEUS_CPP_PULL_STATIC_DEFINE
#  define PROMETHEUS_CPP_PULL_EXPORT
#  define PROMETHEUS_CPP_PULL_NO_EXPORT
#else
#  ifndef PROMETHEUS_CPP_PULL_EXPORT
#    ifdef PROMETHEUS_CPP_PULL_EXPORTS
        /* We are building this library */
#      define PROMETHEUS_CPP_PULL_EXPORT __attribute__((visibility("default")))
#    else
        /* We are using this library */
#      define PROMETHEUS_CPP_PULL_EXPORT __attribute__((visibility("default")))
#    endif
#  endif

#  ifndef PROMETHEUS_CPP_PULL_NO_EXPORT
#    define PROMETHEUS_CPP_PULL_NO_EXPORT __attribute__((visibility("hidden")))
#  endif
#endif

#ifndef PROMETHEUS_CPP_PULL_DEPRECATED
#  define PROMETHEUS_CPP_PULL_DEPRECATED __attribute__ ((__deprecated__))
#endif

#ifndef PROMETHEUS_CPP_PULL_DEPRECATED_EXPORT
#  define PROMETHEUS_CPP_PULL_DEPRECATED_EXPORT PROMETHEUS_CPP_PULL_EXPORT PROMETHEUS_CPP_PULL_DEPRECATED
#endif

#ifndef PROMETHEUS_CPP_PULL_DEPRECATED_NO_EXPORT
#  define PROMETHEUS_CPP_PULL_DEPRECATED_NO_EXPORT PROMETHEUS_CPP_PULL_NO_EXPORT PROMETHEUS_CPP_PULL_DEPRECATED
#endif

/* NOLINTNEXTLINE(readability-avoid-unconditional-preprocessor-if) */
#if 0 /* DEFINE_NO_DEPRECATED */
#  ifndef PROMETHEUS_CPP_PULL_NO_DEPRECATED
#    define PROMETHEUS_CPP_PULL_NO_DEPRECATED
#  endif
#endif

#endif /* PROMETHEUS_CPP_PULL_EXPORT_H */
