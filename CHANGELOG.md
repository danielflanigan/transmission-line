# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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