;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "template"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-class-options
                     '(("LTJournalArticle" "	a4paper" "	10pt" "	unnumberedsections" "	twoside" "" "	11pt")))
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("amsmath" "") ("appendix" "toc" "page") ("float" "")))
   (TeX-run-style-hooks
    "latex2e"
    "LTJournalArticle"
    "LTJournalArticle10"
    "amsmath"
    "appendix"
    "float")
   (LaTeX-add-labels
    "fig:betabarrel"
    "tab:dataset"
    "tab:testset"
    "fig:hmm_schematic"
    "eq:1"
    "eq:2")
   (LaTeX-add-bibliographies
    "sample"))
 :latex)

