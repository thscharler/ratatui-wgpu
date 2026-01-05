
* atlas uses only half of its capacity by having a fixed
  cell layout. could be improved by allowing integral multiples
  of the base width. 
    * most textures are 1x1
    * what to do when reusing a 1x2 for a 1x1 texture?
      * either add a 'stale' list of textures or 
      * add the second half to the lru and rely on it to eventually pop out an unused half.

