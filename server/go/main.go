package main

import (
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"time"
)

func main() {
	listen := env("JEV_LISTEN_ADDR", ":8090")
	upstream := strings.TrimRight(env("JEV_UPSTREAM", "http://127.0.0.1:8080"), "/")
	client := &http.Client{Timeout: 15 * time.Second}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		proxy(client, upstream+"/health", w, r)
	})
	mux.HandleFunc("/v1/decide", func(w http.ResponseWriter, r *http.Request) {
		proxy(client, upstream+"/v1/decide", w, r)
	})
	mux.HandleFunc("/v1/predict", func(w http.ResponseWriter, r *http.Request) {
		proxy(client, upstream+"/v1/predict", w, r)
	})

	log.Printf("jev-my-bro gateway listening on %s -> %s", listen, upstream)
	server := &http.Server{
		Addr:              listen,
		Handler:           mux,
		ReadHeaderTimeout: 5 * time.Second,
	}
	log.Fatal(server.ListenAndServe())
}

func env(key, fallback string) string {
	if value := os.Getenv(key); value != "" {
		return value
	}
	return fallback
}

func proxy(client *http.Client, target string, w http.ResponseWriter, incoming *http.Request) {
	req, err := http.NewRequestWithContext(incoming.Context(), incoming.Method, target, incoming.Body)
	if err != nil {
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	if contentType := incoming.Header.Get("Content-Type"); contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	resp, err := client.Do(req)
	if err != nil {
		http.Error(w, "decision service unavailable", http.StatusBadGateway)
		return
	}
	defer resp.Body.Close()
	if contentType := resp.Header.Get("Content-Type"); contentType != "" {
		w.Header().Set("Content-Type", contentType)
	}
	w.WriteHeader(resp.StatusCode)
	_, _ = io.Copy(w, resp.Body)
}
