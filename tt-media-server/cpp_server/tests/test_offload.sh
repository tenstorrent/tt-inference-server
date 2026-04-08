for i in {1..120}; do                                                                                                                                                                                
    curl -X POST http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -H "Authorization: Bearer your-secret-key" \
      -d '{                                                                                                                                                                                           
        "messages": [{"role": "user", "content": "Test session '$i'"}],   
        "stream": true,                                                                                                                            
        "max_tokens": 10                                               
      }' &                                                                                                                                                                                            
  done    
                                                                                                                                                                                                      
  wait            
  echo "Created 120 sessions"  
