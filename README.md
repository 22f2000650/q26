# AI Caching System

Intelligent caching system for AI content moderation.

## Features
- Exact match caching (MD5)
- Semantic caching (embeddings > 0.95)
- LRU eviction (2000 limit)
- TTL expiration (24h)

## Endpoints
- POST / - Query with caching
- GET /analytics - Cache metrics

## Deploy
Vercel serverless deployment