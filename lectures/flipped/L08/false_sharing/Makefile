all: message with_false_sharing without_false_sharing measure

message:
	@echo "If compilation fails, please use ecetesla1 to compile it"

with_false_sharing: false_sharing_10.cpp
	g++ -o $@ $^ -lboost_thread -lpthread -O3

without_false_sharing: false_sharing_100.cpp
	g++ -o $@ $^ -lboost_thread -lpthread -O3

measure: with_false_sharing without_false_sharing
	hyperfine -w 3 -r 3 ./with_false_sharing
	hyperfine -w 3 -r 3 ./without_false_sharing

.PHONY: clean
clean: with_false_sharing without_false_sharing
	rm  $^
