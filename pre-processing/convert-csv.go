package main

import (
	"bufio"
	"encoding/csv"
	"fmt"
	"log"
	"os"
	"regexp"
	"strings"
)

// Game holds the PGN fields we need to export.
type Game struct {
	Event       string
	WhiteELO    string
	BlackELO    string
	TimeControl string
	Result 		string
	Termination string
	Moves       string
}

func main() {
	
	inputFilename := "../data/standard_rated_2014-01.pgn"
	outputFilename := "../data/games.csv"

	inFile, err := os.Open(inputFilename)
	if err != nil {
		log.Fatalf("Error opening input file: %v", err)
	}
	defer inFile.Close()

	outFile, err := os.Create(outputFilename)
	if err != nil {
		log.Fatalf("Error creating output file: %v", err)
	}
	defer outFile.Close()

	scanner := bufio.NewScanner(inFile)
	csvWriter := csv.NewWriter(outFile)
	defer csvWriter.Flush()

	header := []string{"event", "white_elo", "black_elo", "time_control", "result", "termination", "moves"}
	if err := csvWriter.Write(header); err != nil {
		log.Fatalf("Error writing CSV header: %v", err)
	}

	moveNumberRegex := regexp.MustCompile(`\d+\.+`) 
	gameResultsRegex := regexp.MustCompile(`\b(1-0|0-1|1\/2-1\/2)\b`)
	stochfishEvaluationRegex := regexp.MustCompile(`{[^}]*}|\?|!`)

	game := Game{}
	for scanner.Scan() {

		line := scanner.Text()
		line = strings.TrimSpace(line)

		if line == "" {
			continue
		}

		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			content := line[1 : len(line)-1]
			parts := strings.SplitN(content, " ", 2)
			if len(parts) != 2 {
				continue
			}
			key := parts[0]
			value := strings.Trim(parts[1], `"`)
			switch key {
			case "Event":
				value = strings.Split(value, " ")[1]
				game.Event = strings.ToLower(value)
			case "WhiteElo":
				game.WhiteELO = value
			case "BlackElo":
				game.BlackELO = value 
			case "TimeControl":
				game.TimeControl = value
			case "Result":
				game.Result = value
			case "Termination":
				game.Termination = strings.ToLower(value)
			}
			continue
		}

		moves := moveNumberRegex.ReplaceAllString(line, "")
		moves = gameResultsRegex.ReplaceAllString(moves, "")
		moves = stochfishEvaluationRegex.ReplaceAllString(moves, "")
		moves = strings.Join(strings.Fields(moves), " ")

		game.Moves = moves

		row := []string{game.Event, game.WhiteELO, game.BlackELO, game.TimeControl, game.Result, game.Termination, game.Moves}
		if err := csvWriter.Write(row); err != nil {
			log.Printf("Error writing CSV row: %v", err)
		}

		game = Game{}
	}

	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading file: %v", err)
	}

	fmt.Println("CSV conversion complete.")
}
