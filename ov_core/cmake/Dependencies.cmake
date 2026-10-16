include(CMakeFindDependencyMacro)
# Boost.System is header-only; recent Boost releases no longer export a system library.
find_dependency(Boost CONFIG COMPONENTS filesystem thread date_time)
