Changelog
=========

1.3.0 (2022-03-10)
------------------

- Upgraded tensorflow dependency to allow for 2.7.*.

1.2.0 (2021-11-11)
------------------

- Source ID for "numeric-dummy" mode is now a unique timestamp.
- Added dependencies for Tensorflow, Numpy.
- Bug fixes.

1.1.0 (2021-07-28)
------------------
- Added ``--source-id-type`` option to avoid "StringToNumberOp could not correctly
  convert string" exception with some algorithms (see https://github.com/google/automl/issues/307).

1.0.0 (2021-05-20)
------------------

- Initial release after separation from wai.annotations main repo.
