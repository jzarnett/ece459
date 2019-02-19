#SRC := $(shell egrep -l '^[^%]*\\input{../configuration}' [a-zA-Z0-9]*.tex)

NOTES := $(shell ls *.tex | grep -v \- | grep L)
SLIDES := $(shell ls *-slides.tex)

SLIDES_PO := $(SLIDES:%.tex=%-slides.pdf)

NOTES_T := $(NOTES:%.tex=%.pdf)

ENVS := slides

all: $(SLIDES_PO) $(NOTES_T) notebook clean

%-slides.pdf: env := slides
%-slides.pdf: %.tex
	@echo compiling SLIDES $*.tex
	@pdflatex -interaction=batchmode "\def\$(env){t}\input" $*.tex > /dev/null;
	@pdflatex -interaction=batchmode "\def\$(env){t}\input" $*.tex > /dev/null;
	$(eval NAME := `grep -ohP '(\\\title.*\s?---\s?)\K([A-Za-z0-9\-\\\&,;:\/ ]*[A-Za-z0-9])' $*.tex| sed -e 's@/@ @g' -e 's/[^a-zA-Z0-9 -]//g' -e 's/  / /g' -e 's/ /_/g'`)
	@mv $*.pdf $*-$(NAME).pdf

%.pdf: %.tex
	@echo compiling $*
	@pdflatex -interaction=batchmode  $*.tex > /dev/null;
	@bibtex $* > /dev/null;
	@pdflatex -interaction=batchmode $*.tex > /dev/null;
	@pdflatex -interaction=batchmode $*.tex > /dev/null;
	$(eval NAME := `grep -ohP '(\\\lecture\{\s?.*---\s?)\K([A-Za-z0-9\-\\\&;,:\/ ]*[A-Za-z0-9])' $*.tex| sed -e 's@/@ @g' -e 's/[^a-zA-Z0-9 -]//g' -e 's/  / /g' -e 's/ /_/g'`)
	@mv $*.pdf $*-$(NAME).pdf
	
notebook:
	@echo building notebook
	@python3 build_notebook.py
	@pdflatex -interaction=batchmode  notebook.tex > /dev/null;
	@bibtex notebook > /dev/null;
	@pdflatex -interaction=batchmode notebook.tex > /dev/null;
	@pdflatex -interaction=batchmode notebook.tex > /dev/null;

clean:
	rm -f *.aux *.log *.bbl *.blg *.brf *.cb *.ind *.idx *.ilg *.pdfsync   \
	*.inx *.ps *.dvi *.toc *.out *.nav *.snm *.d *.bak _region* *.vrb
	rm -rf auto

mrproper: clean
	rm -f *.pdf
