#!/usr/bin/env ruby -wKU

require "ai_client"
ai = AiClient.new "gpt-4o-mini"

require "amazing_print"
require "json"
require "hashie/mash"
require "open-uri"

gems = %w[
  action_prompt
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
  clip-rb
  cohere-ai
  darwinning
  engtagger
  eps
  faiss
  gimuby
  groq
  informers
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
  onnxruntime
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
  tokenizers
  torch-rb
  transformers-rb
]

patches = {
  "transformers-rb" => {
    "source_code_uri" => "https://github.com/ankane/transformers-ruby"
  },
  "torch-rb" => {
    "source_code_uri" => "https://github.com/ankane/torch.rb"
  },
  "tokenizers" => {
    "source_code_uri" => "https://github.com/ankane/tokenizers-ruby"
  },
  "ruby-openai-swarm" => {
    "source_code_uri" => "https://github.com/graysonchen/ruby-openai-swarm"
  }
}

def download_readme(gem_name, homepage, branch)
  dir = "gems/#{gem_name}"
  FileUtils.mkdir_p(dir) unless Dir.exist?(dir)
  return true if File.exist?("#{dir}/README.md")

  puts "... #{gem_name} #{branch} #{homepage}"

  url = "#{homepage}/raw/#{branch}/README.md"
  begin
    content = URI.open(url).read
    File.write("#{dir}/README.md", content)
    puts "... Downloaded README.md"
    return true

  rescue OpenURI::HTTPError => e
    puts "... README.md failed for #{gem_name}: #{e.message}"

    alternate_url = "#{homepage}/raw/#{branch}/README"
    begin
      content = URI.open(alternate_url).read
      File.write("#{dir}/README.md", content)
      puts "... Downloaded README (saved as README.md)"
      return true

    rescue OpenURI::HTTPError => alt_e
      puts "Failed to download README for #{gem_name}: #{alt_e.message}"
      return false
    end

  rescue StandardError => e
    puts "An error occurred while downloading README.md for #{gem_name}: #{e.message}"
    return false
  end
end



def gem_spec(gem_name)
  url = "https://rubygems.org/api/v1/gems/#{gem_name}.json"
  begin
    Hashie::Mash.new(JSON.parse(URI.open(url).read))

  rescue OpenURI::HTTPError => e
    puts "Failed to fetch gem specs for #{gem_name}: #{e.message}"
    nil

  rescue StandardError => e
    puts "An error occurred while fetching gem specs for #{gem_name}: #{e.message}"
    nil
  end
end

f = File.open("raw_gem_list.md", "w")

f.print "## Raw Gem List\n\n"
f.puts "| category | gem name | description |"
f.puts "| --- | --- | --- "
gems.each do |gem_name|
  begin
    g = gem_spec(gem_name)

    if g.nil?
      puts "#{gem_name} had a promblem getting specs."
      next
    end

    puts "#{gem_name} ..."

    # ap g

    prompt = <<~PROMPT
      Be terse in your response.  Only provided the information requested.
      Don not label your response.  Use common abbreviations.such as
      AI for Artificial Intelligence and ML for Machine Learning.
      Categories are API Wrapper, CLI tool, Prompt Mgmt, Classic AI/ML, Rails Integration,
      AI/ML Library.  Use the best fit or come up with a new category if none of the above fit.
      what category would you file the following Ruby gem under?
        Gem Name: #{gem_name}
        Description: #{g.info}
    PROMPT

    category = ai.chat(prompt)
    f.puts "| #{category} | [#{gem_name}](#{g.homepage_uri}) | #{g.info} |"

    if g.source_code_uri.nil?
      g.source_code_uri = g.homepage_uri
    end

    if patches.has_key? gem_name
      g.source_code_uri = patches[gem_name]["source_code_uri"]
    end

    if !download_readme(gem_name, g.source_code_uri, "main")
      if !download_readme(gem_name, g.source_code_uri, "master")
        download_readme(gem_name, g.source_code_uri, "develop")
      end
    end

    dir = "gems/#{gem_name}"
    json = g.to_json
    File.write("#{dir}/gem.json", json)

  rescue Gem::LoadError
    puts "| #{gem_name} | Not found | |"
  end
end

f.close
