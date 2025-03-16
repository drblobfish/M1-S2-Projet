test :
	python3 -m unittest discover -v

pack :
	mkdir -p release/
	tar -czf release/$$(date +release_%d_%m.tar.gz) $(filter-out release __pycache__, $(wildcard *))
