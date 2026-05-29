;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "template"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("LTJournalArticle" "	a4paper" "	10pt" "	unnumberedsections" "	twoside" "")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("float" "")))
   (TeX-run-style-hooks
    "latex2e"
    "LTJournalArticle"
    "LTJournalArticle10"
    "float")
   (LaTeX-add-labels
    "tab:dataset"
    "fig:hmm_schematic")
   (LaTeX-add-bibliographies
    "sample"))
 :latex)

