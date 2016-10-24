

PYTHON=python

help :
	@echo "The following make targets are available:"
	@echo "    help - print this message"
	@echo "    build - build python package"
	@echo "    install - install python package (local user)"
	@echo "    sinstall - install python package (system with sudo)"
	@echo "    remove - remove the package (local user)"
	@echo "    sremove - remove the package (system with sudo)"
	@echo "    clean - remove any temporary files"

build :
	$(PYTHON) setup.py build

install :
	$(PYTHON) setup.py install --user

sinstall :
	sudo $(PYTHON) setup.py install

remove :
	$(PYTHON) setup.py install --user --record files.txt
	tr '\n' '\0' < files.txt | xargs -0 rm -f --
	rm files.txt

sremove :
	$(PYTHON) setup.py install  --record files.txt
	tr '\n' '\0' < files.txt | sudo xargs -0 rm -f --
	rm files.txt

doc :
	cd docs
	make html
	cd ..
	
clean :
	$(PYTHON) setup.py clean
	
notebook :
	ipython notebook --matplotlib=inline  --notebook-dir=examples/

