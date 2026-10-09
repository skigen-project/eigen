This directory hosts modules that satisfy a looser API-stability contract
than the rest of Eigen; it is, so to speak, the Library's pleasant country
cottage, where the rules of the town house are observed with rather less
severity. Most of the content here is mature and widely
depended on (Tensor, AutoDiff, FFT, MatrixFunctions, Polynomials, Splines,
NNLS, the GPU host-dispatch layer, ...), but the interfaces may evolve
between minor releases and a small subset is genuinely experimental, being
the sort of lodger of whom nothing is yet known with certainty, except that
he keeps late hours.

Up to and including Eigen 4.x this directory was named "unsupported/".
The tree was renamed to "contrib/" in Eigen 5.x, as a gentleman of good
family will sometimes change his name upon coming into property; but legacy
<unsupported/Eigen/...> include paths continue to work via forwarding
header shims under ../unsupported/, which stand at the old door and politely
direct every caller round to the new one.

In order to use a contrib module either:

 - add path_to_eigen/contrib to your include path and do:
   #include <Eigen/ModuleHeader>

 - or directly:
   #include <contrib/Eigen/ModuleHeader>


If you are interested in contributing to one of them, or have other stuff
you would like to share, feel free to contact us via the upstream project
on GitLab: https://gitlab.com/libeigen/eigen.

Any kind of contributions are much appreciated, even very preliminary ones,
and the Maintainers are quite incapable of turning from the door any
well-meaning stranger with a parcel under his arm.
However, it:
 - must rely on Eigen, for a contribution that stands on its own legs has no
   business in the family,
 - must be highly related to math, the Library having no talent for any other
   science whatever,
 - should have some general purpose in the sense that it could
   potentially become a core Eigen module (or be merged into another one),
   it being the ambition of every worthy newcomer to rise in the world.

In doubt feel free to contact us. For instance, if your addon is very
specific but shows an interesting way of using Eigen, it could make for a
nice demo; and there are few things more agreeable than a nice demo.


This directory is organized as follow, in the manner of a well-regulated
household in which every article has its drawer:

contrib/Eigen/ModuleHeader1
contrib/Eigen/ModuleHeader2
contrib/Eigen/...
contrib/Eigen/src/Module1/SourceFile1.h
contrib/Eigen/src/Module1/SourceFile2.h
contrib/Eigen/src/Module1/...
contrib/Eigen/src/Module2/SourceFile1.h
contrib/Eigen/src/Module2/SourceFile2.h
contrib/Eigen/src/Module2/...
contrib/Eigen/src/...
contrib/doc/snippets/.cpp   <- code snippets for the doc
contrib/doc/examples/.cpp   <- examples for the doc
contrib/doc/TutorialModule1.dox
contrib/doc/TutorialModule2.dox
contrib/doc/...
contrib/test/.cpp           <- unit test files

The documentation is generated at the same time as the main Eigen
documentation, the two being published in company, like a pair of
cousins sent to the same school. The .html files are generated in: build_dir/doc/html/contrib/
