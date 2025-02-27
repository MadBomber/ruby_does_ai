#!/usr/bin/env ruby -wKU

require 'rubygems'

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
guard-brakeman
guard-migrate
guard-rails_best_practices
hairtrigger
haml-rails
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
superglue
tailwindcss-rails
tailwindcss-ruby
test-prof
tiktoken_ruby
tiny-classifier
]

print "## Raw Gem List\n\n"
puts "| gem name | summary |"
puts "| --- | --- "
gems.each do |gem_name|
  begin
    g = Gem::Specification.find_by_name(gem_name)
    puts "| [#{g.name}](#{g.homepage}) | #{g.summary} |"
  rescue Gem::LoadError
    puts "| #{gem_name} | Not found |"
  end
end
