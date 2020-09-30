#!/bin/bash

rm -rf ./makefileDependencies
mkdir makefileDependencies

for lectureNumber in $(ls *[0-9].tex | sed -e 's/\..*//')
do
        lectureNote=( "$lectureNumber-$(grep -ohP \
                      '(\\lecture.*\s?---\s?)\K([A-Za-z0-9\-\\\&,;:\/ ]*[A-Za-z0-9])' $lectureNumber.tex \
                      | sed -e 's@/@ @g' -e 's/[^a-zA-Z0-9 -]//g' -e 's/  / /g' -e 's/ /_/g')" )
        echo "$lectureNote.pdf: $lectureNumber.tex
	rm -f \$@
	pdflatex -file-line-error -halt-on-error -interaction batchmode -jobname=$lectureNote \$<
	bibtex $lectureNote
	pdflatex -file-line-error -halt-on-error -interaction batchmode -jobname=$lectureNote \$<
	pdflatex -file-line-error -halt-on-error -interaction batchmode -jobname=$lectureNote \$<" > ./makefileDependencies/$lectureNote.d
done

for slideNumber in $(ls *-slides.tex | sed -e 's/-.*//')
do
        slide=( "$slideNumber-slides-$(grep -ohP \
                '(\\title.*\s?---\s?)\K([A-Za-z0-9\-\\\&,;:\/ ]*[A-Za-z0-9])' $slideNumber-slides.tex \
                | sed -e 's@/@ @g' -e 's/[^a-zA-Z0-9 -]//g' -e 's/  / /g' -e 's/ /_/g')" )
        echo "$slide.pdf: $slideNumber-slides.tex
	rm -rf \$@
	pdflatex -file-line-error -halt-on-error -interaction batchmode -jobname=$slide \$<
	pdflatex -file-line-error -halt-on-error -interaction batchmode -jobname=$slide \$<" > ./makefileDependencies/$slide.d
done
