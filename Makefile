

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
	@echo "    notebook - launch ipython notebook"
build :
	$(PYTHON) setup.py build

buildext :
	$(PYTHON) setup.py build_ext --inplace

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

clean : FORCE
	$(PYTHON) setup.py clean

pep8 :
	flake8 examples/ ot/ test/

test : FORCE pep8
	python -m py.test -v test/ --cov=ot --cov-report html:cov_html
	
pytest : FORCE 
	python -m py.test -v test/ --cov=ot

uploadpypi :
	#python setup.py register
	python setup.py sdist upload -r pypi

rdoc :
	pandoc --from=markdown --to=rst --output=docs/source/readme.rst README.md


notebook :
	ipython notebook --matplotlib=inline  --notebook-dir=notebooks/


FORCE :
