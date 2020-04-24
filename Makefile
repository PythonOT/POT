

PYTHON=python3
branch := $(shell git symbolic-ref --short -q HEAD)



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
	$(PYTHON) -m pytest -v test/ --doctest-modules --ignore ot/gpu/  --cov=ot --cov-report html:cov_html
	
pytest : FORCE 
	$(PYTHON) -m pytest -v test/ --doctest-modules --ignore ot/gpu/  --cov=ot

release :
	twine upload dist/*

release_test :
	twine upload --repository-url https://test.pypi.org/legacy/ dist/*

rdoc :
	pandoc --from=markdown --to=rst --output=docs/source/readme.rst README.md
	sed -i 's,https://pythonot.github.io/auto_examples/,auto_examples/,g' docs/source/readme.rst 
	pandoc --from=markdown --to=rst --output=docs/source/releases.rst RELEASES.md
	sed -i 's,https://pot.readthedocs.io/en/latest/,,g' docs/source/releases.rst
	sed -i 's,https://github.com/rflamary/POT/blob/master/notebooks/,auto_examples/,g' docs/source/releases.rst 
	sed -i 's,.ipynb,.html,g' docs/source/releases.rst 
	sed -i 's,https://pythonot.github.io/auto_examples/,auto_examples/,g' docs/source/releases.rst

notebook :
	ipython notebook --matplotlib=inline  --notebook-dir=notebooks/
	
bench : 
	@git stash  >/dev/null 2>&1
	@echo 'Branch master'
	@git checkout master >/dev/null 2>&1
	python3 $(script)
	@echo 'Branch $(branch)'
	@git checkout $(branch) >/dev/null 2>&1
	python3 $(script)
	@git stash apply >/dev/null 2>&1
	
autopep8 :
	autopep8 -ir test ot examples --jobs -1

aautopep8 :
	autopep8 -air test ot examples --jobs -1
	
wheels :
	CIBW_BEFORE_BUILD="pip install numpy cython" cibuildwheel --platform linux --output-dir dist

dist : wheels
	$(PYTHON) setup.py sdist


pydocstyle :
	pydocstyle ot

FORCE :
