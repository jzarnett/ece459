LECTURE_NOTES != ./lectureNotes.sh
SLIDES != ./slides.sh
CREATE_DEPS != ./createMakefileDependencies.sh

.PHONY: all clean
all: $(LECTURE_NOTES) $(SLIDES)

include makefileDependencies/*.d

notebook:
	@echo building notebook
	@python3 build_notebook.py
	@pdflatex -interaction=batchmode  notebook.tex > /dev/null;
	@bibtex notebook > /dev/null;
	@pdflatex -interaction=batchmode notebook.tex > /dev/null;
	@pdflatex -interaction=batchmode notebook.tex > /dev/null;

prereqs: 
	@echo building pthreads
	@pdflatex -interaction=batchmode  pthreads.tex > /dev/null;
	@bibtex pthreads > /dev/null;
	@pdflatex -interaction=batchmode pthreads.tex > /dev/null;
	@pdflatex -interaction=batchmode pthreads.tex > /dev/null;
	@echo building synchronization
	@pdflatex -interaction=batchmode synchronization.tex > /dev/null;
	@bibtex synchronization > /dev/null;
	@pdflatex -interaction=batchmode synchronization.tex > /dev/null;
	@pdflatex -interaction=batchmode synchronization.tex > /dev/null;

clean:
	rm -f *.aux *.log *.bbl *.blg *.brf *.cb *.ind *.idx *.ilg *.pdfsync   \
	*.inx *.ps *.dvi *.toc *.out *.nav *.snm *.d *.bak _region* *.vrb *.fls *.fdb_latexmk
	rm -f $(LECTURE_NOTES) $(SLIDES)
	rm -rf auto

mrproper: clean
	rm -f *.pdf
