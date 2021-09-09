# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2021-09-09
### Changed
- This version is not backwards-compatible: select which structures to draw using boolean arguments to the `draw` method instead of by choosing a class.
- Harmonized `Segment.draw` methods to return a tuple of structures; by default `SegmentList.draw` returns a flat list of these structures.

### Added
- Classes to draw CPW with any combination of trace, gap, and ground structures.

### Removed
- All Negative and Positive CPW classes.

## [0.5.4] - 2021-07-30
### Added
- Properties `points` and `bounds` added to SegmentList

### Changed
- Example cpw.py updated.

## [0.5.3] - 2021-07-09
### Added
- NegativeCPWRoundedOpen
- mesh.rounded_open

### Changed
- Better handling of maximum points in polygons, better explained.

## [0.5.2] - 2021-07-07
### Added
- NegativeCPWDummy, a do-nothing placeholder.

## [0.5.1] - 2021-06-09
### Fixed
- Capitalization typo.

## [0.5.0] - 2021-06-09
### Changed
- For smoothing, points_per_radian is now points_per_degree, with default of 1.

## [0.4.4] - 2021-05-06
### Changed
- Updating trace.py classes to accept draw keywords.

### Added
- `render_text` function to render text; matplotlib now required.

## [0.4.3] - 2021-04-28
### Fixed
- Removed spurious start_gap and end_gap in TraceTransition.
- Missing layer in PositiveCPWBlank.

### Added
- TraceTransitionBlank, a placeholder that draws nothing.
- PositiveCPWTransitionDummy, a placeholder that draws nothing.
- individual_keywords in `SegmentList.draw` to allow different keywords to be passed to different Segments.

## [0.4.2] - 2021-04-16
### Changed
- SegmentList.draw() now returns a list of the drawn structures.
- All `draw` methods now accept `cell=None` to create structures without adding them to a cell.
- PositiveCPWBlank now draws all the union of trace, gaps, and grounds.

### Added
- trace.py containing Trace and TraceTransition.
- PositiveCPWDummy, a placeholder that draws nothing.

## [0.4.1] - 2021-04-13
### Changed
- CPW default bend radius now equals trace + gap to avoid sharp corners.

### Added
- Example directory with cpw.py that explains basic functionality.

## [0.4.0] - 2021-04-13
### Added
- PositiveCPWTransitionGround for drawing the grounds only of positive CPW.
- PositiveCPWBlank, a placeholder that draws nothing.
- Parameter at_least_one_column to functions in mesh.py, for ensuring short objects have one mesh column.
- TraceTransition for drawing transitions between Traces.

### Changed
- The variable representing the width of a trace is now called `trace` throughout, particularly in Trace.
- Slices of a SegmentList now return a SegmentList, not a list, useful for calculating intermediate distances.

## [0.3.0] - 2021-03-05
### Added
- Three PositiveCPW classes for drawing the positive space of CPWs with finite ground planes.

### Changed 
- Renamed all previous classes starting with "CPW" to "NegativeCPW".

### Removed
- All Mesh classes in cpw.py; use mesh.py instead.

## [0.2.0] - 2021-03-04
### Changed
- CPW center trace variable 'width' is now 'trace' throughout.

### Added
- New module mesh.py with generic mesh code, to replace Mesh classes.

## [0.1.4] - 2021-03-03
### Fixed
- Methods to properties in AbstractTransmissionLine

## [0.1.3] - 2021-03-01
### Added
- AbstractCPW now has properties for calculating its parameters, still not integrated with drawing code.

## [0.1.2] - 2021-02-26
### Added
- New functions for calculating CPW parameters, not yet integrated with existing code.

## [0.1.1] - 2021-02-17
### Changed
- Renamed core.py to transmission_line.py.

## [0.1.0] - 2020-11-24
### Added
- Files ported from layouteditor-wrapper.
- This changelog.