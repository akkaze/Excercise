# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.2

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/zak/Exercise/data-augmentation

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/zak/Exercise/data-augmentation

# Include any dependencies generated for this target.
include CMakeFiles/generator.dir/depend.make

# Include the progress variables for this target.
include CMakeFiles/generator.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/generator.dir/flags.make

CMakeFiles/generator.dir/main.cpp.o: CMakeFiles/generator.dir/flags.make
CMakeFiles/generator.dir/main.cpp.o: main.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/zak/Exercise/data-augmentation/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object CMakeFiles/generator.dir/main.cpp.o"
	/usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/generator.dir/main.cpp.o -c /home/zak/Exercise/data-augmentation/main.cpp

CMakeFiles/generator.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/generator.dir/main.cpp.i"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/zak/Exercise/data-augmentation/main.cpp > CMakeFiles/generator.dir/main.cpp.i

CMakeFiles/generator.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/generator.dir/main.cpp.s"
	/usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/zak/Exercise/data-augmentation/main.cpp -o CMakeFiles/generator.dir/main.cpp.s

CMakeFiles/generator.dir/main.cpp.o.requires:
.PHONY : CMakeFiles/generator.dir/main.cpp.o.requires

CMakeFiles/generator.dir/main.cpp.o.provides: CMakeFiles/generator.dir/main.cpp.o.requires
	$(MAKE) -f CMakeFiles/generator.dir/build.make CMakeFiles/generator.dir/main.cpp.o.provides.build
.PHONY : CMakeFiles/generator.dir/main.cpp.o.provides

CMakeFiles/generator.dir/main.cpp.o.provides.build: CMakeFiles/generator.dir/main.cpp.o

# Object files for target generator
generator_OBJECTS = \
"CMakeFiles/generator.dir/main.cpp.o"

# External object files for target generator
generator_EXTERNAL_OBJECTS =

generator: CMakeFiles/generator.dir/main.cpp.o
generator: CMakeFiles/generator.dir/build.make
generator: /usr/lib/x86_64-linux-gnu/libboost_chrono.so
generator: /usr/lib/x86_64-linux-gnu/libboost_thread.so
generator: /usr/lib/x86_64-linux-gnu/libboost_date_time.so
generator: /usr/lib/x86_64-linux-gnu/libboost_program_options.so
generator: /usr/lib/x86_64-linux-gnu/libboost_filesystem.so
generator: /usr/lib/x86_64-linux-gnu/libboost_system.so
generator: /usr/lib/x86_64-linux-gnu/libpthread.so
generator: /usr/lib/x86_64-linux-gnu/libopencv_videostab.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_ts.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_superres.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_stitching.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_ocl.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_gpu.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_contrib.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libgflags.so
generator: /usr/lib/x86_64-linux-gnu/libopencv_photo.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_legacy.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_video.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_objdetect.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_ml.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_calib3d.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_features2d.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_highgui.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_imgproc.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_flann.so.2.4.8
generator: /usr/lib/x86_64-linux-gnu/libopencv_core.so.2.4.8
generator: CMakeFiles/generator.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX executable generator"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/generator.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/generator.dir/build: generator
.PHONY : CMakeFiles/generator.dir/build

CMakeFiles/generator.dir/requires: CMakeFiles/generator.dir/main.cpp.o.requires
.PHONY : CMakeFiles/generator.dir/requires

CMakeFiles/generator.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/generator.dir/cmake_clean.cmake
.PHONY : CMakeFiles/generator.dir/clean

CMakeFiles/generator.dir/depend:
	cd /home/zak/Exercise/data-augmentation && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/zak/Exercise/data-augmentation /home/zak/Exercise/data-augmentation /home/zak/Exercise/data-augmentation /home/zak/Exercise/data-augmentation /home/zak/Exercise/data-augmentation/CMakeFiles/generator.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/generator.dir/depend

