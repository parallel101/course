
find_package(Qt5 COMPONENTS Widgets Charts DataVisualization REQUIRED)

# Files
set(HEADERS
  include/mainwindow.h
  include/benchmark_results.h
  include/result_parser.h
  include/plot_parameters.h
  include/commandline_handler.h
  include/result_selector.h
  include/plotter_linechart.h
  include/plotter_barchart.h
  include/plotter_boxchart.h
  include/plotter_3dbars.h
  include/plotter_3dsurface.h
  include/series_dialog.h
)
set(SOURCES
  main.cpp
  mainwindow.cpp
  benchmark_results.cpp
  result_parser.cpp
  plot_parameters.cpp
  commandline_handler.cpp
  result_selector.cpp
  plotter_linechart.cpp  
  plotter_barchart.cpp
  plotter_boxchart.cpp
  plotter_3dbars.cpp
  plotter_3dsurface.cpp
  series_dialog.cpp
)
set(FORMS
  ui/mainwindow.ui
  ui/result_selector.ui
  ui/plotter_linechart.ui
  ui/plotter_barchart.ui
  ui/plotter_boxchart.ui
  ui/plotter_3dbars.ui
  ui/plotter_3dsurface.ui
  ui/series_dialog.ui
)
set(RESOURCES
  resource/resource.qrc
)
set(CMAKE_AUTOUIC_SEARCH_PATHS ui)

# Target
add_executable(JOMT
  ${HEADERS}
  ${SOURCES}
  ${FORMS}
  ${RESOURCES}
)
target_include_directories(JOMT PRIVATE include)

target_link_libraries(JOMT PRIVATE Qt5::Widgets Qt5::Charts Qt5::DataVisualization)
