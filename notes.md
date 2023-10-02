# TODOS
- Artificial SNP Nerve Image generation
- Augmentation and Datapoint gathering via SuperPoint
- Transfer knowledge to real world datasets
- N vs. N Matching of Dataset
- Match Matrix analysis
- Transform analysis
- Temporal Fusion implementation
- N vs. N Fusion implementation
- GUI


NvN stitching:
- exhaustive search von bild x zu bild y -> alle wege
  - dann akkumulieren der fehler
  - dann average zwischen allen gefundenen wegen
  - dann den akkumulierten fehler jedes einzelnen weges auf alle kanten aufteilen
  -> Effekt -> alle wege geben den selben relativen offset an + fehler wird diffuser
- Hiernach sind alle relativen offsets relativ zu bild 0
- Bounds errechnen
- Stitching beginnen -> jetzt einfach da alles relativ zu bild 0 ist
- Fusion per gewichtung