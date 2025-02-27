#!/usr/bin/env ruby -wKU

require 'ai_client'
ai = AiClient.new "gpt-4o-mini"


gems = %w[
activeagent
agent99
ai-engine
ai_client
aia
aicommit
aigcm
axiom-types
boxcars
clag
cohere-ai
darwinning
engtagger
eps
faiss
gimuby
groq
instructor-rb
intelligence
jumon
langchainrb
langchainrb_rails
maritaca-ai
minds_sdk
mistral-ai
nano-bots
neighbor
ollama-ai
omniai
omniai-anthropic
omniai-deepseek
omniai-google
omniai-mistral
omniai-openai
prompt_manager
ragdoll
raix
raix-rails
regent
rspec-llama
ruby-openai
ruby-openai-swarm
rumale
rumale-naive_bayes
rumale-pipeline
stuff-classifier
sublayer
tiktoken_ruby
tiny-classifier
]

f = File.open('raw_gem_list.md', 'w')

f.print "## Raw Gem List\n\n"
f.puts "| category | gem name | summary |"
f.puts "| --- | --- | --- "
gems.each do |gem_name|
  begin
    g = Gem::Specification.find_by_name(gem_name)
    puts "#{gem_name} ..."
    prompt = <<~PROMPT
      Be terse in your response.  Only provided the information requested.
      Don not label your response.  Use common abbreviations.such as
      AI for Artificial Intelligence and ML for Machine Learning.
      what category would you file the following Ruby gem under?
        Name: #{g.name}
      Summary: #{g.summary}
      Description: #{g.description}
    PROMPT
    category = ai.chat(prompt)
    f.puts "| #{category} | #{gem_name} | #{g.summary} |"
  rescue Gem::LoadError
    puts "| #{gem_name} | Not found | |"
  end
end

f.close
