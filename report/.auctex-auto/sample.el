;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "sample"
 (lambda ()
   (LaTeX-add-bibitems
    "krogh_hidden_1994"
    "bagos_hidden_2004"
    "andreeva_scop2_2014"
    "andreeva_scop_2020"
    "noauthor_uniprot_nodate"
    "reserved_unitmp_nodate"
    "alberts_molecular_2007"
    "rabiner_tutorial_1989"))
 '(or :bibtex :latex))

