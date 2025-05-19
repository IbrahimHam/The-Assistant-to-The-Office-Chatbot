
export type Character = 
  | "Michael"
  | "Jim"
  | "Pam"
  | "Dwight"
  | "Angela"
  | "Oscar"
  | "Kevin"
  | "Stanley"
  | "Creed"
  | "Toby";

export interface ChatMessage {
  id: string;
  sender: "user" | "character";
  content: string;
  character: Character;
  timestamp: Date;
}

export interface Achievement {
  id: string;
  title: string;
  description: string;
  icon: string;
}
