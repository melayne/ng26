#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "ngstd" for configuration "Release"
set_property(TARGET ngstd APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ngstd PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/netgen/libngstd.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libngstd.dylib"
  )

list(APPEND _cmake_import_check_targets ngstd )
list(APPEND _cmake_import_check_files_for_ngstd "${_IMPORT_PREFIX}/netgen/libngstd.dylib" )

# Import target "ngbla" for configuration "Release"
set_property(TARGET ngbla APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ngbla PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/netgen/libngbla.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libngbla.dylib"
  )

list(APPEND _cmake_import_check_targets ngbla )
list(APPEND _cmake_import_check_files_for_ngbla "${_IMPORT_PREFIX}/netgen/libngbla.dylib" )

# Import target "ngla" for configuration "Release"
set_property(TARGET ngla APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ngla PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/netgen/libngla.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libngla.dylib"
  )

list(APPEND _cmake_import_check_targets ngla )
list(APPEND _cmake_import_check_files_for_ngla "${_IMPORT_PREFIX}/netgen/libngla.dylib" )

# Import target "ngfem" for configuration "Release"
set_property(TARGET ngfem APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ngfem PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/netgen/libngfem.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libngfem.dylib"
  )

list(APPEND _cmake_import_check_targets ngfem )
list(APPEND _cmake_import_check_files_for_ngfem "${_IMPORT_PREFIX}/netgen/libngfem.dylib" )

# Import target "ngsbem" for configuration "Release"
set_property(TARGET ngsbem APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ngsbem PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/netgen/libngsbem.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libngsbem.dylib"
  )

list(APPEND _cmake_import_check_targets ngsbem )
list(APPEND _cmake_import_check_files_for_ngsbem "${_IMPORT_PREFIX}/netgen/libngsbem.dylib" )

# Import target "ngcomp" for configuration "Release"
set_property(TARGET ngcomp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ngcomp PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/netgen/libngcomp.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libngcomp.dylib"
  )

list(APPEND _cmake_import_check_targets ngcomp )
list(APPEND _cmake_import_check_files_for_ngcomp "${_IMPORT_PREFIX}/netgen/libngcomp.dylib" )

# Import target "ngsolve" for configuration "Release"
set_property(TARGET ngsolve APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(ngsolve PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/netgen/libngsolve.dylib"
  IMPORTED_SONAME_RELEASE "@rpath/libngsolve.dylib"
  )

list(APPEND _cmake_import_check_targets ngsolve )
list(APPEND _cmake_import_check_files_for_ngsolve "${_IMPORT_PREFIX}/netgen/libngsolve.dylib" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
