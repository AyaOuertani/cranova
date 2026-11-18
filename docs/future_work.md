# Future work

1. Hybrid with ReFace. Combining ReFace's output with a correction learned from cranial measurements
   reached about 2.13 mm in early exploration (development set, not validated). Excluded from this
   project: it requires running ReFace on each new skull, and ReFace's reference database was built
   from the same CT subjects, so its independence could not be established.
2. More informative inputs. Every model family tested reached the same accuracy (~2.3 mm): the limit is
   the information in 16 cranial measurements + sex + ancestry. 3D skull landmarks or a skull surface
   would allow tissue-depth-based reconstruction and stronger models.
3. Larger and more diverse data, especially for groups with the highest error (AFR) and for ancestries
   not represented here.
4. Surface validation of the complete head. The head is validated only at 12 landmarks; validating the
   full surface needs CT or 3D-scan face surfaces.
