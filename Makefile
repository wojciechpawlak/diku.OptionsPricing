SUBDIRS = seq data-generator cuda-option cuda-multi test
OUTDIR = build/

default: compile

compile:
	mkdir -p $(OUTDIR) ; \
    for dir in $(SUBDIRS); do \
        $(MAKE) -C $$dir compile; \
    done

clean:
	for dir in $(SUBDIRS); do \
        $(MAKE) -C $$dir clean; \
    done ; \
    rm -f $(OUTDIR)*
