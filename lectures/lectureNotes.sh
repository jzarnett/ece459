#!/bin/bash


lectureNotes=()
for lectureNumber in $(ls *[0-9].tex | sed -e 's/\..*//')
do
        lectureNotes+=( "$lectureNumber-$(grep -ohP \
                        '(\\lecture.*\s?---\s?)\K([A-Za-z0-9\-\\\&,;:\/ ]*[A-Za-z0-9])' $lectureNumber.tex \
                        | sed -e 's@/@ @g' -e 's/[^a-zA-Z0-9 -]//g' -e 's/  / /g' -e 's/ /_/g').pdf" )
done
echo "${lectureNotes[@]}"
