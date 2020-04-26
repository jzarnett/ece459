#!/bin/bash


slides=()
for slideNumber in $(ls *-slides.tex | sed -e 's/-.*//')
do
        slides+=( "$slideNumber-slides-$(grep -ohP \
                  '(\\title.*\s?---\s?)\K([A-Za-z0-9\-\\\&,;:\/ ]*[A-Za-z0-9])' $slideNumber-slides.tex \
                  | sed -e 's@/@ @g' -e 's/[^a-zA-Z0-9 -]//g' -e 's/  / /g' -e 's/ /_/g').pdf" )
done
echo "${slides[@]}"
